import logging
import sys
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from collections import defaultdict
from model import FineTuneModel
from utils.encoders import CJPreprocessCollator
import datasets
from utils.training import get_param_norm, get_grad_norm, count_parameters, move_to
from utils.config import training_config, get_model_config

import torchmetrics as tm

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")

#accelerator = Accelerator(log_with="wandb")


config = training_config(sys.argv[1])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(config.seed)

dataset = datasets.load_from_disk(config.dataset).with_format('torch')
dataset = dataset.map(lambda sample: {'labels': sample['targets'][config.loss_config.class_index]})
#Presplit and stuff

# Collator
model_config = get_model_config(config)
device = accelerator.device
print(config.loss_config)
model = FineTuneModel( config.run_predictor,
                        model_config['embedding'],
                        model_config['encoder'],
                        model_config['predictor'],
                        config.decoder,
                        config.loss_config)

preprocessing_collator = CJPreprocessCollator(num_mask = config.num_mask,
        transform = config.transform,
        rotate = config.rotate,
        smiles_col = config.smiles_col,
        encoder = config.encoder.type)

config.n_params_emb, config.n_params_nonemb = count_parameters(model, print_summary=False)

# Initialise your wandb run, passing wandb parameters and any config information
init_kwargs={"wandb": {"entity": "josiahbjorgaard"}}
accelerator.init_trackers(
    project_name=config.project_name,
    config=dict(config),
    init_kwargs=init_kwargs
    )

# Creating a DataLoader object for iterating over it during the training epochs
train_dl = DataLoader( dataset["train"],
                       batch_size=config.batch_size,
                       shuffle=True,
                       collate_fn = preprocessing_collator,
                       num_workers=8,
                       prefetch_factor=16)
eval_dl = DataLoader( dataset["test"],
                      batch_size=config.batch_size,
                      collate_fn=preprocessing_collator,
                      )

accelerator.print(f"Number of embedding parameters: {config.n_params_emb/10**6}M")
accelerator.print(f"Number of non-embedding parameters: {config.n_params_nonemb/10**6}M")
accelerator.print(f"Number of training samples: {len(dataset['train'])}")
accelerator.print(f"Number of training batches per epoch: {len(train_dl)}")

num_training_steps = config.epochs * len(train_dl)

optimizer = AdamW(model.parameters(), lr=config.lr) # * world_size)
lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

if accelerator.is_main_process:
    progress_bar = tqdm(range(num_training_steps), initial = config.start_epoch * len(train_dl))

logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
     model, optimizer, train_dl, eval_dl, lr_scheduler
     )

# Start model training and defining the training loop
if config.loss_config.type.lower() == "bce":
    metrics = {
        'precision': tm.Precision(task='binary'),
        'recall': tm.Recall(task='binary'),
        'accuracy': tm.Accuracy(task='binary'),
        'cm': tm.ConfusionMatrix(task='binary'),
        'f1': tm.F1Score(task='binary'),
        'specificity': tm.Specificity(task='binary'),
        'auroc': tm.AUROC(task='binary'),    
        'auprc': tm.AveragePrecision(task='binary'),
    }
else:
    metrics = {
        'pcc':  tm.PearsonCorrCoef(),
        'mae': tm.MeanAbsoluteError(),
        'mse': tm.MeanSquaredError(),
        }

for v in metrics.values():
    v.to(accelerator.device)

world_size = torch.cuda.device_count()
for epoch in range(config.start_epoch,config.epochs):
    model.train()
    for idb, batch in tqdm(enumerate(train_dl)):
        # Training
        batch, xbatch, xmask, metadata = batch #preprocessing(batch)
        labels = torch.FloatTensor(metadata['labels'])
        labels = move_to(labels, device)
        batch = move_to(batch, device)
        outputs = model(labels, batch)
        optimizer.zero_grad()
        loss, logits = outputs #['loss']
        #loss = outputs
        for k,v in metrics.items():
            if k not in ['pcc', 'mse', 'mae']:
                labels = labels.to(torch.long)
            v.update(logits.detach(), labels.detach())
        accelerator.backward(loss)
        if config.clip:
            accelerator.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()
        lr_scheduler.step()

        # Log and checkpoint
        #if idb % config.n_step_checkpoint == 0:
        #    accelerator.save_state(config.output_dir)
        if accelerator.is_main_process:
            progress_bar.update(world_size)

        accelerator.log({"logit_mean": logits.flatten().mean().detach().to("cpu"),
                         "logit_var": logits.flatten().var().detach().to("cpu")})
        accelerator.log({"total_loss": loss.detach().to("cpu")})
        accelerator.log({"param_norm": get_param_norm(model).to("cpu"),
                         "grad_norm": get_grad_norm(model).to("cpu")})
        accelerator.log({"lr": optimizer.param_groups[0]['lr']})
    accelerator.log({k:v.compute() for k,v in metrics.items()})
    for v in metrics.values():
        v.reset()
    #Epoch end log and checkpoint
    #os.makedirs(os.path.join(config.output_dir,str(epoch)), exist_ok=True)
    #accelerator.save_state(os.path.join(config.output_dir, str(epoch)))

    #Eval looop
    if config.run_eval_loop:
        model.eval()
        with torch.no_grad():
            losses = defaultdict(lambda: torch.Tensor([0.0]).to("cpu"))
            true = []
            pred = []
            sample_index = []
            for i, batch in enumerate(tqdm(eval_dl)):
                batch, xbatch, xmask, metadata = batch  # preprocessing(batch)
                labels = torch.FloatTensor(metadata['labels'])
                labels = move_to(labels, device) 
                batch = move_to(batch, device)
                outputs = model(labels, batch)
                loss, logits = outputs #['loss']
                for k,v in metrics.items():
                    if k not in ['pcc', 'mse', 'mae']:
                        labels = labels.to(torch.long)
                    v.update(logits.detach(), labels.detach())
                losses["total_loss"] += loss.detach().to("cpu")
                accelerator.log({"val_step_total_loss":loss.to("cpu")})
                #sample_index+=batch['sample_index'].detach().cpu().flatten().tolist()
                true.append(labels.detach().to('cpu').to(torch.long))
                pred.append(logits.detach().to('cpu'))
            pred = torch.cat(pred).flatten()
            true = torch.cat(true).flatten()
            accelerator.log({'val_epoch_'+k: v/len(eval_dl) for k, v in losses.items()})
            accelerator.log({'val_logit_mean': pred.mean(), 'logit_var': pred.var()})
            accelerator.log({"val_"+k: v.compute() for k, v in metrics.items()})
            for v in metrics.values():
                v.reset()
            #file_path = os.path.join(config.output_dir,f"{str(epoch)}_{accelerator.process_index}_preds.csv")
            #pd.DataFrame.from_dict({
            #        'sample_index':sample_index,
            #        'true':true.tolist(),
            #        'pred':pred.tolist()}).to_csv(file_path)
logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

accelerator.save_model(model, config.output_dir, safe_serialization=True)
accelerator.end_training()
