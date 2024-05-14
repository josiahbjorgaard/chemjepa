from yacs.config import CfgNode as CN
from datetime import datetime
import os
from contextlib import redirect_stdout
import json
import yaml


def get_cfg_defaults_train():
    """
    Default config options for training
    """
    config = CN(new_allowed=True)
    config.label_col = "Labels"
    config.restart = "" #'training_output_21_31_23_10_2023'
    config.wandb_name = "No Name"
    config.wandb_restart = ""
    config.epochs = 3
    config.start_epoch = 0
    config.batch_size = 2
    config.num_warmup_steps = 3000
    config.lr_scheduler_type = "cosine"
    config.lr = 1e-4
    config.weight_decay = 0.04
    config.reset_lr = False
    config.output_dir = "" #datetime.now().strftime('training_output_%H_%M_%d_%m_%Y')
    config.dataset = "/shared/dataset3M" #"/shared/fcaa53cd-ba57-4bfe-af9c-eaa958f95c1a_combined_all"
    config.split = 0.1
    config.ds_frac = 1.0
    config.ds_seed = 42
    config.seed = 42
    config.dropout = 0.1
    config.clip = 0.0
    config.n_step_checkpoint = 20000
    config.run_eval_loop = True

    config.num_mask = 4
    config.transform = ""
    config.rotate = "all"
    
    config.smiles_col = "SMILES"

    config.embedding = CN(new_allowed=True)  # None #{}
    config.embedding.pad_len = 1024
    config.embedding.num_embeddings = 450
    config.embedding.padding_token = 0

    config.encoder = CN(new_allowed=True) #None #{}
    config.encoder.hidden_size = 512  # hidden size
    config.encoder.layers = 12  # layers
    config.encoder.heads = 4  # num heads
    config.encoder.dim_head = 128  # heads * dim_head = intermediate size
    config.encoder.ff_mult = 4  # Feed forward multiplier
    config.encoder.ema_decay = 0.998
    config.encoder.freeze_layers = 0
    config.encoder.type = ""
    config.predictor = CN(new_allowed=True)  # None #{}
    config.predictor.hidden_size = 384  # hidden size
    config.predictor.layers = 12  # layers
    config.predictor.heads = 4  # num heads
    config.predictor.dim_head = 96  # heads * dim_head = intermediate size
    config.predictor.ff_mult = 4  # Feed forward multiplier
    config.predictor.fine_tune_with_class_token = False
    config.predictor.freeze_layers = 0
    config.decoder = CN(new_allowed=True)
    return config.clone()

def restart_cfg(config):
    """
    Revise config options if restarting
    """
    if config.restart:
        # Allow creating new keys recursively.
        config.set_new_allowed(True)
        config.merge_from_file(os.path.join(config.restart, 'config.yaml'))
        config.epochs = 1 ### WILL NEED TO SPECIFY NUMBER OF EPOCHS TO CONTINUE WITH HERE
        ### New Output directory!!
        config.output_dir = datetime.now().strftime('training_output_%H_%M_%d_%m_%Y')
        config.reset_lr = 0.0001
    return config

def training_config(filename):
    config = get_cfg_defaults_train()
    with open(filename, "r") as stream:
        config_dict = yaml.safe_load(stream)
    new_config = CN(config_dict)
    if not config.output_dir:
        output_dir = datetime.now().strftime('training_output_%H_%M_%d_%m_%Y')
        config.output_dir = output_dir
        i = 1
        while os.path.isdir(config.output_dir):
            config.output_dir = output_dir + f'_{i}' 
            i+=1 
    print(new_config)
    config.merge_from_other_cfg(new_config)
    dump_configs(config, config.output_dir)
    return config


def get_model_config(config):
    #### MODEL
    model_config = {
        'embedding': dict(config.embedding),
        'encoder': dict(config.encoder),
        'predictor': dict(config.predictor)
    }
    return model_config


def dump_configs(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir,'config.yaml'),'w') as f:
        with redirect_stdout(f): print(config.dump())

def dump_model_configs(config, output_dir):
    with open(os.path.join(output_dir,'model_config.json'),'w') as f:
        json.dump(get_model_config(config), f)
