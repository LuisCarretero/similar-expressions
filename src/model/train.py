from model import LitGVAE
from config_util import load_config
import lightning as L
from data_util import create_dataloader, calc_priors_and_means
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb

seed_everything(42, workers=True, verbose=False)


def main(cfg_path, data_path, dataset_name):
    # Load config
    cfg_dict, cfg = load_config(cfg_path)

    # Load data
    train_loader, valid_loader, info = create_dataloader(data_path, dataset_name, cfg)
    priors, means = calc_priors_and_means(train_loader)  # TODO: Introduce bias initialization

    # Setup model
    gvae = LitGVAE(cfg, priors)

    # Setup logger
    logger = WandbLogger(project='similar-expressions-01')
    cfg_dict['dataset_hashes'] = info['hashes']
    cfg_dict['dataset_name'] = info['dataset_name']
    logger.log_hyperparams(cfg_dict)

    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb.run.dir, 
        filename='{epoch:02d}', 
        monitor='valid/loss', 
        mode='min', 
        save_top_k=1, 
        save_last=True
    )

    # Train model
    trainer = L.Trainer(
        logger=logger, 
        max_epochs=cfg.training.epochs, 
        gradient_clip_val=cfg.training.optimizer.clip,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(gvae, train_loader, valid_loader)


if __name__ == '__main__':
    cfg_path = '/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/src/model/config.json'
    data_path = '/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data'
    main(cfg_path, data_path, dataset_name='dataset_240817_2')




