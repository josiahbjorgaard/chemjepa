import logging
import os
import sys
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from transformers import get_scheduler
from model_torchmd import TensorNet
from utils.training import get_param_norm, get_grad_norm, count_parameters, move_to
from utils.config import training_config, get_model_config
#ifrom utils.dataset import setup_data
from utils.dataset import PairsDataset
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
"""
datasets = setup_data(config.dataset,
                      split=config.split,
                      ds_frac=config.ds_frac,
                      ds_seed=config.ds_seed)

"""# Collator
model_config = get_model_config(config)
device = accelerator.device

# Model
xenc_model = TensorNet() #**model_config['tensornet'])
decay = model_config['encoder']['ema_decay']
yenc_model = AveragedModel(xenc_model, multi_avg_fn=get_ema_multi_avg_fn(decay))
#yenc_model = xenc_model
pred_model = TensorNet() #**model_config['tensornet'])
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

# Dataset from TensorNet
dataset = PairsDataset("../notebooks/ani1x_pairs_filtered",mask=0.2)

# Creating a DataLoader object for iterating over it during the training epochs
train_dl = DataLoader(dataset, batch_size=config.batch_size,
                       shuffle=True, num_workers=8, prefetch_factor=4,
                       )
#eval_dl = DataLoader(datasets["test"], batch_size=config.batch_size,
#                       num_workers=8, prefetch_factor=4)

accelerator.print(f"Number of encoder embedding parameters: {config.encoder_n_params_emb/10**6}M")
accelerator.print(f"Number of encoder non-embedding parameters: {config.encoder_n_params_nonemb/10**6}M")
accelerator.print(f"Number of predictor non-embedding parameters: {config.predictor_n_params_nonemb/10**6}M")
accelerator.print(f"Number of training samples: {len(dataset)}")
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

xenc_model, yenc_model, pred_model, optimizer, train_dl,  lr_scheduler, loss_function  = accelerator.prepare(
     xenc_model, yenc_model, pred_model, optimizer, train_dl,  lr_scheduler, loss_function
     )

#yenc_model.to(accelerator.device)

# Start model training and defining the training loop


world_size = torch.cuda.device_count()
for epoch in range(config.start_epoch, config.epochs):
    xenc_model.train()
    pred_model.train()
    yenc_model.eval()
    for idb, batch in tqdm(enumerate(train_dl)):
        #TODO pairing and masking should be done previously
        #batch, xbatch = move_to(batch, device), move_to(xbatch, device)

        xX, edge_index, edge_weight, edge_attr, q = xenc_model.module.pre_forward(batch['z'],batch['pos'], 
                  batch['batch'], batch['label'], batch['mask'])
        x = xenc_model(xX, edge_index, edge_weight, edge_attr, q) #x in the context
        
        with torch.no_grad():
            yX, edge_index, edge_weight, edge_attr, q = yenc_model.module.module.pre_forward(batch['z'],batch['pos2'],
                    batch['batch'], batch['label2'], None)
            y = yenc_model(yX, edge_index, edge_weight, edge_attr, q) #Target Encoder
            y = yenc_model.module.module.post_forward(y)

        x = pred_model(x, edge_index, edge_weight, edge_attr, q) #pred model - use target geometry with context embeddings
        x = pred_model.module.post_forward(x)

        mask = batch['mask'] # atoms with masked forces in context encoder
        loss = loss_function(x[mask], y[mask]) #Loss is only for masked tokens
        if torch.isnan(loss):
            print(f"{loss=}")
            for k,v in batch.items():
                if torch.isnan(v).sum():
                    print(k)
                    print(torch.isnan(v).sum())
        optimizer.zero_grad()
        accelerator.backward(loss)
        if config.clip:
            accelerator.clip_grad_norm_(nn.ModuleList([xenc_model,pred_model]).parameters(), config.clip)

        optimizer.step()
        lr_scheduler.step()
        yenc_model.module.update_parameters(xenc_model)
        # Log and checkpoint
        #if idb % config.n_step_checkpoint == 0:
            #accelerator.save_state(config.output_dir)
        if accelerator.is_main_process:
            progress_bar.update(world_size)

        accelerator.log({"total_loss": loss.detach().to("cpu"),
                         #"xenc_var": enc_var,
                         #"pred_var": pred_var,
                         #"yenc_var": y_var,
                         #'xenc_mean': enc_mean,
                         #'pred_mean': pred_mean,
                         #'yenc_mean': y_mean,
                         "xenc_param_norm": get_param_norm(xenc_model).to("cpu"),
                         "xenc_grad_norm": get_grad_norm(xenc_model).to("cpu"),
                         "pred_param_norm": get_param_norm(pred_model).to("cpu"),
                         "pred_grad_norm": get_grad_norm(pred_model).to("cpu"),
                         "lr": optimizer.param_groups[0]['lr']})
    #Epoch end log and checkpoint
    os.makedirs(os.path.join(config.output_dir, str(epoch)), exist_ok=True)
    accelerator.save_state(os.path.join(config.output_dir, str(epoch)))

logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
accelerator.save_model(xenc_model, config.output_dir + "xenc_model", safe_serialization=True)
accelerator.save_model(pred_model, config.output_dir + "pred_model", safe_serialization=True)
accelerator.end_training()
