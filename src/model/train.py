import torch
from model import GrammarVAE
from util import AnnealKL, AnnealKLSigmoid, create_dataloader, load_config, criterion_factory, calc_syntax_accuracy
import wandb
from tqdm import tqdm
import hashlib

def train_one_epoch(train_loader, epoch_idx: int):
    log_accumulators = {
        'loss': 0, 'loss_syntax_onehot': 0, 'loss_syntax_consts': 0,
        'loss_value': 0, 'kl': 0, 'alpha': 0, 'elbo': 0, 'syntax_accuracy': 0
    }
    log_steps = 0

    for step, (x, y_rule_idx, y_consts, y_val) in tqdm(enumerate(train_loader, 1), desc=f'Epoch {epoch_idx}/{config["epochs"]}', total=len(train_loader)):
        mu, sigma = model.encoder(x)
        z = model.sample(mu, sigma)
        logits = model.decoder(z, max_length=config['seq_len'])
        values = model.value_decoder(z)
        
        loss_syntax_onehot, loss_syntax_consts, loss_value, loss = criterion(logits, values, y_rule_idx, y_consts, y_val)
        kl = model.kl(mu, sigma)
        alpha = anneal.alpha(epoch_idx)   # FIXME: Step vs epoch
        elbo = loss + alpha*kl

        optimizer.zero_grad()
        elbo.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
        optimizer.step()
        assert not any(torch.isnan(param).any() for param in model.parameters()), "NaN found in model parameters"

        syntax_accuracy = calc_syntax_accuracy(logits, y_rule_idx)
        for key, value in zip(log_accumulators.keys(), [loss, loss_syntax_onehot, loss_syntax_consts, loss_value, kl, alpha, elbo, syntax_accuracy]):
            log_accumulators[key] += value.item() if isinstance(value, torch.Tensor) else value
        log_steps += 1

        if step % config['print_every'] == 0:
            avg_logs = {f'train/{k}': v / log_steps for k, v in log_accumulators.items()}
            wandb.log(avg_logs)
            log_accumulators = {k: 0 for k in log_accumulators}
            log_steps = 0

    if log_steps > 0:
        avg_logs = {f'train/{k}': v / log_steps for k, v in log_accumulators.items()}
        wandb.log(avg_logs)

def test(test_loader, epoch_idx: int):
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
            elbo = loss + anneal.alpha(epoch_idx) * kl
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
    model = GrammarVAE(config['encoder_hidden'], config['z_size'], config['conv_size'], config['decoder_hidden'], config['token_cnt'], config['rnn_type'], config['val_points'], config['device'])

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    anneal = AnnealKLSigmoid(total_epochs=config['epochs'], midpoint=0.3, steepness=10)

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
    train_loader, test_loader = create_dataloader(datapath, 
                                                  'dataset_240817_2', 
                                                  test_split=1/4, 
                                                  batch_size=config['batch_size'], 
                                                  value_transform=value_transform, 
                                                  device=config['device'],
                                                  max_length=None)


    # hash_string = hashlib.md5(str([train_loader.dataset[i][3].std().item() for i in range(1000)]).encode()).hexdigest()
    # print(f'Hash string: {hash_string}')
    # exit()

    for epoch in range(1, config['epochs']+1):
        train_one_epoch(train_loader, epoch)
        test(test_loader, epoch)

    # torch.save(model, f'{wandb.run.dir}/model.pt')
    torch.save({'model_state_dict': model.state_dict()}, f'{wandb.run.dir}/model.pth')
