from model import LitGVAE
from config_util import load_config
import lightning as L
from data_util import create_dataloader, calc_priors_and_means
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb

seed_everything(42, workers=True, verbose=False)


def main():
    cfg_path = '/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/src/model/config.json'
    data_path = '/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data'

    # Load config
    cfg_dict, cfg = load_config(cfg_path)

    # Load data
    train_loader, test_loader, info = create_dataloader(data_path, 'dataset_240817_2', cfg)  # FIXME: Rename test -> valid
    priors, means = calc_priors_and_means(train_loader)

    # Setup model
    gvae = LitGVAE(cfg, priors)

    # Setup logger
    logger = WandbLogger(project='similar-expressions-01')
    cfg_dict['dataset_hashes'] = info['hashes']
    cfg_dict['dataset_name'] = info['dataset_name']
    logger.log_hyperparams(cfg_dict)

    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb.run.dir,
        filename='gvae-{epoch:02d}',
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
    trainer.fit(gvae, train_loader, test_loader)
    trainer.save_checkpoint("final_model.ckpt")


if __name__ == '__main__':
    main()




