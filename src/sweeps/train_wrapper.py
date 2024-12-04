import wandb
from src.model.util import load_config, update_cfg
from src.model.train import train_model

def exec_single_run():
    wandb.init(dir='/store/DAMTP/lc865/workspace/wandb-cache')
    assert len(wandb.config.keys()) > 0, "Wandb config is empty. Script needs to be called via wandb agent."

    # Load default config
    cfg = load_config('src/model/config.yaml')

    # Update default config with sweep parameters (partial config with parameters that are sweeped over)
    update_cfg(cfg, wandb.config)

    # Run training
    data_path = ['/store/DAMTP/lc865/workspace/data', '/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/data'][0]
    train_model(cfg, data_path, dataset_name='dataset_241204_2', overwrite_device_count=1, overwrite_strategy='auto')  # FIXME: overwrite no longer needed?

if __name__ == '__main__':
    exec_single_run()