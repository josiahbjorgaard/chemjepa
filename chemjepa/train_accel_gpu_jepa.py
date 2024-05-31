import logging
import os
import sys
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from model import CJEncoder, HFEncoder, CJPredictor, make_predictor_tokens
from utils.encoders import CJPreprocessCollator
from utils.training import get_param_norm, get_grad_norm, count_parameters, move_to
from utils.config import training_config, get_model_config
from utils.dataset import setup_data
from accelerate import Accelerator
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
torch.autograd.set_detect_anomaly(True)
from accelerate import DistributedDataParallelKwargs
from safetensors.torch import load_model
torch.autograd.set_detect_anomaly(True)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
#accelerator = Accelerator(log_with="wandb")

config = training_config(sys.argv[1])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(config.seed)

datasets = setup_data(config.dataset,
                      split=config.split,
                      ds_frac=config.ds_frac,
                      ds_seed=config.ds_seed)

# Collator
model_config = get_model_config(config)
device = accelerator.device

# Model
if model_config['encoder']['type']=='chemberta':
    xenc_model = HFEncoder(**model_config['encoder'], embedding_config=model_config['embedding'])
else:
    xenc_model = CJEncoder(**model_config['encoder'], embedding_config=model_config['embedding'])
decay = model_config['encoder']['ema_decay']
yenc_model = AveragedModel(xenc_model, multi_avg_fn=get_ema_multi_avg_fn(decay))
#yenc_model = xenc_model
pred_model = CJPredictor(**model_config['predictor'])
config.encoder_n_params_emb, config.encoder_n_params_nonemb = count_parameters(xenc_model, print_summary=False)
config.predictor_n_params_emb, config.predictor_n_params_nonemb = count_parameters(pred_model, print_summary=False)

# Initialise your wandb run, passing wandb parameters and any config information
init_kwargs={"wandb": {"entity": "josiahbjorgaard"}}
if config.wandb_restart:
    init_kwargs["wandb"]["id"]=config.wandb_restart
    init_kwargs["wandb"]["resume"]="must"
accelerator.init_trackers(
    project_name="ChemJEPA",
    config=dict(config),
    init_kwargs=init_kwargs
    )

preprocessing_collator = CJPreprocessCollator(num_mask = config.num_mask,
        transform = config.transform,
        rotate = config.rotate,
        encoder = config.encoder.type)

# Creating a DataLoader object for iterating over it during the training epochs
train_dl = DataLoader( datasets["train"], batch_size=config.batch_size,
                       shuffle=True, num_workers=8, prefetch_factor=4,
                       collate_fn = preprocessing_collator)
eval_dl = DataLoader( datasets["test"], batch_size=config.batch_size,
                      collate_fn = preprocessing_collator, num_workers=8, prefetch_factor=4)

accelerator.print(f"Number of encoder embedding parameters: {config.encoder_n_params_emb/10**6}M")
accelerator.print(f"Number of encoder non-embedding parameters: {config.encoder_n_params_nonemb/10**6}M")
accelerator.print(f"Number of predictor non-embedding parameters: {config.predictor_n_params_nonemb/10**6}M")
accelerator.print(f"Number of training samples: {len(datasets['train'])}")
accelerator.print(f"Number of training batches per epoch: {len(train_dl)}")

num_training_steps = config.epochs * len(train_dl)

optimizer = AdamW(nn.ModuleList([xenc_model,pred_model]).parameters(), lr=config.lr, weight_decay = config.weight_decay)
lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

if accelerator.is_main_process:
    progress_bar = tqdm(range(num_training_steps), initial = config.start_epoch * len(train_dl))

logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

loss_function = nn.MSELoss() #ContrastiveLossWithTemperature()


if config.restart:
    logger.info(f"Loading saved state from {config.restart}")
    load_model(xenc_model, os.path.join(config.restart,'model.safetensors'))
    load_model(yenc_model, os.path.join(config.restart,'model_1.safetensors'))
    load_model(pred_model, os.path.join(config.restart, 'model_2.safetensors'))
    #load_model(optimizer, os.path.join(config.restart, 'model_3.safetensors'))
    #accelerator.load_state(config.restart)
    if config.reset_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.reset_lr

xenc_model, yenc_model, pred_model, optimizer, train_dl, eval_dl, lr_scheduler, loss_function  = accelerator.prepare(
     xenc_model, yenc_model, pred_model, optimizer, train_dl, eval_dl, lr_scheduler, loss_function
     )

#yenc_model.to(accelerator.device)

# Start model training and defining the training loop

xenc_model.train()
pred_model.train()
yenc_model.eval()

world_size = torch.cuda.device_count()
for epoch in range(config.start_epoch,config.epochs):
    for idb, batch in tqdm(enumerate(train_dl)):
        # Mutation and masking function here
        batch, xbatch, xmask, _ = batch #preprocessing(batch)
        batch, xbatch, xmask = move_to(batch, device), move_to(xbatch, device), move_to(xmask, device)
        # Training
        x, (tokens, attention_mask) = xenc_model(xbatch, batch) #(tokens, attention_mask) are the prediction tokens
        enc_var = torch.var(x.detach(), dim=0).mean().cpu()
        x = torch.cat([x, tokens], dim=1)
        attention_mask = torch.cat([xbatch['attention_mask'], attention_mask * 2], dim=1)
        if (~torch.isfinite(attention_mask)).sum():
            print(attention_mask)
            raise Exception('Nan value in mask')
        elif (~torch.isfinite(x)).sum():
            print(x)
            raise Exception("Nan value in x")
        x = pred_model(x, attention_mask.to(torch.bool))
        pred_var = torch.var(x.detach(), dim=0).mean().cpu()
        xmask = attention_mask == 2
        ymask = batch['target_mask']
        
        with torch.no_grad():
            y = yenc_model(batch) #Target Encoder gets all context and all tokens
            y_var = torch.var(y, dim=0).mean().cpu()
        loss = loss_function(x[xmask], y[ymask]) #Loss is only for masked tokens
        optimizer.zero_grad()
        accelerator.backward(loss)
        if config.clip:
            accelerator.clip_grad_norm_(nn.ModuleList([xenc_model,pred_model]).parameters(), config.clip)

        optimizer.step()
        lr_scheduler.step()
        yenc_model.module.update_parameters(xenc_model)
        #yenc_model.update_parameters(xenc_model)
        # Log and checkpoint
        #if idb % config.n_step_checkpoint == 0:
            #accelerator.save_state(config.output_dir)
        if accelerator.is_main_process:
            progress_bar.update(world_size)

        accelerator.log({"total_loss": loss.detach().to("cpu"),
                         "xenc_var": enc_var,
                         "pred_var": pred_var,
                         "yenc_var": y_var,
                         "xenc_param_norm": get_param_norm(xenc_model).to("cpu"),
                         "xenc_grad_norm": get_grad_norm(xenc_model).to("cpu"),
                         "pred_param_norm": get_param_norm(pred_model).to("cpu"),
                         "pred_grad_norm": get_grad_norm(pred_model).to("cpu"),
                         "lr": optimizer.param_groups[0]['lr']})
    #Epoch end log and checkpoint
    #os.makedirs(os.path.join(config.output_dir, str(epoch)), exist_ok=True)
    #accelerator.save_state(os.path.join(config.output_dir, str(epoch)))
    #Eval looop
    if config.run_eval_loop:
        xenc_model.eval()
        pred_model.eval()
        with torch.no_grad():
            epoch_loss = 0.0
            for i, batch in enumerate(tqdm(eval_dl)):
                       # Mutation and masking function here
                batch, xbatch, xmask, _ = batch
                batch, xbatch, xmask = move_to(batch, device), move_to(xbatch, device), move_to(xmask, device)
                # Training
                x, (tokens, attention_mask) = xenc_model(xbatch, batch) #, xmask)
                x = torch.cat([x, tokens],dim=1)
                attention_mask = torch.cat([xbatch['attention_mask'], attention_mask * 2], dim=1)
                x = pred_model(x, attention_mask.to(torch.bool))
                xmask = attention_mask == 2
                ymask = batch['target_mask']
                y = yenc_model(batch) #Target Encoder gets all context and all tokens
                loss = loss_function(x[xmask], y[ymask]) #Loss is only for masked tokens

                #Step Log
                accelerator.log({"val_step_loss":loss.to("cpu")})
                epoch_loss += loss.to("cpu")
            #Epoch Log
            accelerator.log({'val_epoch_loss': epoch_loss})

logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
accelerator.save_model(xenc_model, config.output_dir + "xenc_model", safe_serialization=True)
accelerator.save_model(pred_model, config.output_dir + "pred_model", safe_serialization=True)
accelerator.end_training()
