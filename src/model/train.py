import torch
from model import GrammarVAE
from util import AnnealKL, create_dataloader, load_config, criterion_factory, calc_syntax_accuracy
import wandb
from tqdm import tqdm

def train_one_epoch(train_loader, epoch_idx: int):
    # Iterate over the training data
    for step, (x, y_rule_idx, y_consts, y_val) in tqdm(enumerate(train_loader, 1), desc=f'Epoch {epoch_idx}/{config["epochs"]}', total=len(train_loader)):
        # Forward pass
        mu, sigma = model.encoder(x)
        z = model.sample(mu, sigma)
        logits = model.decoder(z, max_length=config['seq_len'])
        values = model.value_decoder(z)
        
        # Calculate losses
        loss_syntax_onehot, loss_syntax_consts, loss_value, loss = criterion(logits, values, y_rule_idx, y_consts, y_val)

        kl = model.kl(mu, sigma)
        alpha = anneal.alpha(step)
        elbo = loss + alpha*kl

        # Update parameters
        optimizer.zero_grad()
        elbo.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
        optimizer.step()
        assert not any(torch.isnan(param).any() for param in model.parameters()), "NaN found in model parameters"

        # Logging
        syntax_accuracy = calc_syntax_accuracy(logits, y_rule_idx)
        if step % config['print_every'] == 0:
            wandb.log({'train/loss': loss, 'train/loss_syntax_onehot': loss_syntax_onehot, 'train/loss_syntax_consts': loss_syntax_consts, 'train/loss_value': loss_value, 'train/kl': kl, 'train/alpha': alpha, 'train/elbo': elbo, 'train/syntax_accuracy': syntax_accuracy})

def test(test_loader):
    metrics = {
        'loss_syntax_onehot': 0, 'loss_syntax_consts': 0, 'loss_value': 0,
        'kl': 0, 'elbo': 0, 'loss': 0, 'syntax_accuracy': 0
    }

    with torch.no_grad():
        for step, (x, y_rule_idx, y_consts, y_val) in tqdm(enumerate(test_loader, 1), desc='Test', total=len(test_loader)):
            mu, sigma = model.encoder(x)
            z = model.sample(mu, sigma)
            logits = model.decoder(z, max_length=config['seq_len'])
            values = model.value_decoder(z)
            
            loss_syntax_onehot, loss_syntax_consts, loss_value, loss = criterion(logits, values, y_rule_idx, y_consts, y_val)
            kl = model.kl(mu, sigma)
            elbo = loss + anneal.alpha(step) * kl
            syntax_accuracy = calc_syntax_accuracy(logits, y_rule_idx)

            for key, value in zip(metrics.keys(), [loss_syntax_onehot, loss_syntax_consts, loss_value, kl, elbo, loss, syntax_accuracy]):
                metrics[key] += value

    for key in metrics:
        metrics[key] /= len(test_loader)

    wandb.log({f'test/{key}': value for key, value in metrics.items()})


if __name__ == '__main__':
    config = load_config('/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/src/model/hyperparameters.json')

    # Init WandB
    run = wandb.init(project="similar-expressions-01", config=config)

    # Init model
    torch.manual_seed(41)
    model = GrammarVAE(config['encoder_hidden'], config['z_size'], config['decoder_hidden'], config['token_cnt'], config['rnn_type'], config['val_points'], config['device'])

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    anneal = AnnealKL(step=1e-3, rate=500)

    # Init loss function
    priors = {
        'syntax_prior': 1.3984073,
        'const_prior': 0.06433585,
        'value_prior': 0.06500606
    }
    criterion = criterion_factory(config, priors)

    # Load data
    def value_transform(x):
        return torch.arcsinh(x)*0.1  # Example transformation. TODO: adjust scaling dynamically (arcsinh(1e5)=12.2 so currently this gives us 1.22)
    datapath = '/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data'
    train_loader, test_loader = create_dataloader(datapath, 'dataset_240817_2', test_split=0.1, batch_size=config['batch_size'], value_transform=value_transform, device=config['device'])

    for epoch in range(1, config['epochs']+1):
        train_one_epoch(train_loader, epoch)
        test(test_loader)

    # torch.save(model, f'{wandb.run.dir}/model.pt')
    torch.save({'model_state_dict': model.state_dict()}, f'{wandb.run.dir}/model.pth')
