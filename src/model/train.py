from model import LitGVAE
from config_util import load_config
import lightning as L
from data_util import create_dataloader, calc_priors_and_means, summarize_dataloaders
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import pickle

seed_everything(42, workers=True, verbose=False)

def main(cfg_path, data_path, dataset_name):
    # Load config
    cfg_dict, cfg = load_config(cfg_path)

    # Load data
    train_loader, valid_loader, info = create_dataloader(data_path, dataset_name, cfg, num_workers=2)
    priors, means = calc_priors_and_means(train_loader)  # TODO: Introduce bias initialization
    summarize_dataloaders(train_loader, valid_loader)

    # Setup model
    gvae = LitGVAE(cfg, priors)

    # Setup logger
    logger = WandbLogger(project='similar-expressions-01')  # Disable automatic syncing)
    cfg_dict['dataset_hashes'] = info['hashes']
    cfg_dict['dataset_name'] = info['dataset_name']
    logger.log_hyperparams(cfg_dict)

    # Save cfg object to wandb files
    # with open(wandb.run.dir + '/config.pkl', 'wb') as f:
    #     pickle.dump(cfg, f)
    # wandb.save('config.pkl')

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=wandb.run.dir, 
    #     filename='{epoch:02d}', 
    #     monitor='valid/loss', 
    #     mode='min', 
    #     save_top_k=0, 
    #     save_last=True
    # )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", 
        min_delta=0.00, 
        patience=4, 
        verbose=False, 
        mode="min"
    )

    # Train model
    trainer = L.Trainer(
        logger=logger, 
        max_epochs=cfg.training.epochs, 
        gradient_clip_val=cfg.training.optimizer.clip,
        callbacks=[early_stopping_callback],
        # profiler=AdvancedProfiler(dirpath='.', filename='profile.txt'),
        log_every_n_steps=100
    )
    trainer.fit(gvae, train_loader, valid_loader)
    wandb.finish()


if __name__ == '__main__':
    cfg_path = '/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/src/model/config.json'
    data_path = '/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data'
    main(cfg_path, data_path, dataset_name='dataset_240817_2')




