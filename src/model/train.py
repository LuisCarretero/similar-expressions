import torch
from model import GrammarVAE
from util import AnnealKLSigmoid, criterion_factory, calc_syntax_accuracy, compute_latent_metrics
from data_util import calc_priors_and_means, create_dataloader
from config_util import load_config
import wandb
from tqdm import tqdm

def train_one_epoch(train_loader, epoch_idx: int):
    metrics = {}
    log_steps = 0

    for step, (x, y_syntax, y_consts, y_values) in tqdm(enumerate(train_loader, 1), desc=f'Epoch {epoch_idx}/{cfg.training.epochs}', total=len(train_loader)):
        # Forward pass
        mean, ln_var = model.encoder(x)
        z = model.sample(mean, ln_var)
        logits = model.decoder(z, max_length=cfg.model.io_format.seq_len)
        values = model.value_decoder(z)
        
        # Compute losses
        kl = model.calc_kl(mean, ln_var)  # Positive definite scalar, aim to minimize
        alpha = anneal.alpha(epoch_idx)
        loss, partial_losses = criterion(logits, values, y_syntax, y_consts, y_values, kl, alpha)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.optimizer.clip)
        optimizer.step()
        assert not any(torch.isnan(param).any() for param in model.parameters()), "NaN found in model parameters"

        # Compute metrics   
        latent_metrics = compute_latent_metrics(mean, ln_var)
        syntax_accuracy = calc_syntax_accuracy(logits, y_syntax)

        # Merge and sum up metrics
        step_metrics = {**partial_losses, **latent_metrics, 'syntax_accuracy': syntax_accuracy}
        for key, value in step_metrics.items():
            if key not in metrics:
                metrics[key] = 0
            metrics[key] += value.item() if isinstance(value, torch.Tensor) else value
        log_steps += 1

        if step % cfg.training.log_interval == 0:
            avg_logs = {f'train/{k}': v / log_steps for k, v in metrics.items()}
            wandb.log(avg_logs)
            metrics = {k: 0 for k in metrics}
            log_steps = 0

    if log_steps > 0:
        avg_logs = {f'train/{k}': v / log_steps for k, v in metrics.items()}
        wandb.log(avg_logs)

def test(test_loader, epoch_idx: int):
    metrics = {}

    with torch.no_grad():
        for step, (x, y_syntax, y_consts, y_values) in tqdm(enumerate(test_loader, 1), desc='Test', total=len(test_loader)):
            # Forward pass
            mean, ln_var = model.encoder(x)
            z = model.sample(mean, ln_var)
            logits = model.decoder(z, max_length=cfg.model.io_format.seq_len)
            values = model.value_decoder(z)

            # Compute losses
            kl = model.calc_kl(mean, ln_var)
            alpha = anneal.alpha(epoch_idx)
            _, partial_losses = criterion(logits, values, y_syntax, y_consts, y_values, kl, alpha)

            # Compute metrics
            syntax_accuracy = calc_syntax_accuracy(logits, y_syntax)
            latent_metrics = compute_latent_metrics(mean, ln_var)

            # Merge and sum up metrics
            step_metrics = {**partial_losses, **latent_metrics, 'syntax_accuracy': syntax_accuracy}
            for key, value in step_metrics.items():
                if key not in metrics:
                    metrics[key] = 0
                metrics[key] += value.item() if isinstance(value, torch.Tensor) else value

    # Calculate average
    for key in metrics:
        metrics[key] /= len(test_loader)

    wandb.log({f'test/{key}': value for key, value in metrics.items()})


if __name__ == '__main__':
    cfg_dict, cfg = load_config('/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/src/model/config.json')

    # Init model
    torch.manual_seed(42)
    model = GrammarVAE(cfg)

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.optimizer.lr)
    anneal = AnnealKLSigmoid(cfg)

    # Load data
    datapath = '/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data'
    train_loader, test_loader, info = create_dataloader(datapath, 'dataset_240822_1', cfg)

    # Init loss function (given priors) and means
    priors, means = calc_priors_and_means(train_loader)
    criterion = criterion_factory(cfg, priors)
    if cfg.training.values_init_bias:
        with torch.no_grad():
            model.value_decoder.fc3.bias.data = means['values_mean']
            # FIXME: Check if LSTM decoder bias can also be initialized?

    # Init WandB
    cfg_dict['dataset_hashes'] = info['hashes']
    cfg_dict['dataset_name'] = info['dataset_name']
    run = wandb.init(project="similar-expressions-01", config=cfg_dict)

    for epoch in range(1, cfg.training.epochs+1):
        train_one_epoch(train_loader, epoch)
        test(test_loader, epoch)

    # torch.save(model, f'{wandb.run.dir}/model.pt')
    torch.save({'model_state_dict': model.state_dict()}, f'{wandb.run.dir}/model.pth')
    run.finish()
