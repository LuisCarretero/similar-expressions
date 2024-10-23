from config_util import load_config
import wandb
from sweep_config import sweep_configuration
from train import train_model


def exec_single_run():
    wandb.init()

    # Load default config
    cfg_dict, cfg = load_config('src/model/config.json')

    # Update config with sweep parameters
    cfg.model.encoder.depth = wandb.config['model.encoder.depth']
    cfg.model.encoder.width = wandb.config['model.encoder.width']
    cfg.model.value_decoder.depth = wandb.config['model.value_decoder.depth']
    cfg.model.value_decoder.width = wandb.config['model.value_decoder.width']
    cfg.model.z_size = wandb.config['model.z_size']
    cfg.training.batch_size = wandb.config['training.batch_size']
    cfg_dict['model']['encoder']['depth'] = wandb.config['model.encoder.depth']
    cfg_dict['model']['encoder']['width'] = wandb.config['model.encoder.width']
    cfg_dict['model']['value_decoder']['depth'] = wandb.config['model.value_decoder.depth']
    cfg_dict['model']['value_decoder']['width'] = wandb.config['model.value_decoder.width']
    cfg_dict['model']['z_size'] = wandb.config['model.z_size']
    cfg_dict['training']['batch_size'] = wandb.config['training.batch_size']

    # Run training
    data_path = '/store/DAMTP/lc865/workspace/data'
    train_model(cfg_dict, cfg, data_path, dataset_name='dataset_241008_1')  # , overwrite_device_count=1, overwrite_strategy='auto'

if __name__ == '__main__':
    exec_single_run()