from config_util import load_config
import wandb
from sweep_config import sweep_configuration
from train import train_model

def sweep_agent():
    wandb.init()
    
    cfg_dict, cfg = load_config('src/model/config.json')

    # Update cfg with sweep parameters
    # cfg.model.encoder.architecture = wandb.config['model.encoder.architecture']
    # cfg.model.decoder.architecture = wandb.config['model.decoder.architecture']
    # cfg.model.value_decoder.architecture = wandb.config['model.value_decoder.architecture']
    # cfg.model.decoder.depth = wandb.config['model.decoder.depth']
    # cfg.model.decoder.width = wandb.config['model.decoder.width']
    cfg.model.value_decoder.depth = wandb.config['model.value_decoder.depth']
    cfg.model.value_decoder.width = wandb.config['model.value_decoder.width']
    cfg.model.encoder.depth = wandb.config['model.encoder.depth']
    cfg.model.encoder.width = wandb.config['model.encoder.width']
    cfg.model.z_size = wandb.config['model.z_size']
    

    # Run your main training function
    data_path = '/store/DAMTP/lc865/workspace/data'
    train_model(cfg_dict, cfg, data_path, dataset_name='dataset_241008_1')

def do_sweep():
    sweep_id = wandb.sweep(sweep_configuration, project="sweep-test-02")  # Initialize sweep
    wandb.agent(sweep_id, function=sweep_agent, count=30)  # Run 10 experiments

if __name__ == '__main__':
    do_sweep()