from config_util import load_config
import wandb
from sweep_config import sweep_configuration
from train import train_model

def sweep_agent():
    wandb.init()
    
    cfg_dict, cfg = load_config('src/model/config.json')

    # Update cfg with sweep parameters
    cfg.model.decoder.rnn_type = wandb.config['model.decoder.rnn_type']
    cfg.model.decoder.hidden_size = wandb.config['model.decoder.hidden_size']
    
    data_path = '/store/DAMTP/lc865/workspace/data'

    # Run your main training function
    train_model(cfg_dict, cfg, data_path, dataset_name='dataset_241008_1')

def do_sweep():
    sweep_id = wandb.sweep(sweep_configuration, project="sweep-test-01")  # Initialize sweep
    wandb.agent(sweep_id, function=sweep_agent, count=5)  # Run 10 experiments

if __name__ == '__main__':
    do_sweep()