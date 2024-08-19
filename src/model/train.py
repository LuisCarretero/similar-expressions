import torch
from model import GrammarVAE
from util import AnnealKL, create_dataloader
import wandb
from tqdm import tqdm
import numpy as np

ENCODER_HIDDEN = 20
Z_SIZE = 10
DECODER_HIDDEN = 20
RNN_TYPE = 'lstm'
BATCH_SIZE = 64
SEQ_LEN = 15  # TODO: Get from dataset
TOKEN_CNT = 10  # TODO: Get from dataset
VAL_POINTS = 100
LR = 4e-3
CLIP = 5.
PRINT_EVERY = 200
EPOCHS = 10
VALUE_LOSS_WEIGHT = 0
CONST_LOSS_WEIGHT = 0
SYNTAX_LOSS_WEIGHT = 1
DEVICE = 'mps'


def calc_syntax_accuracy(logits, y_rule_idx):
    y_hat = logits.argmax(-1)
    a = (y_hat == y_rule_idx).float().mean()
    return 100 * a.item()


def train_one_epoch(train_loader, epoch_idx: int):
    # Iterate over the training data
    for step, (x, y_rule_idx, y_consts, y_val) in tqdm(enumerate(train_loader, 1), desc=f'Epoch {epoch_idx}/{EPOCHS}', total=len(train_loader)):
        # Forward pass
        mu, sigma = model.encoder(x)
        z = model.sample(mu, sigma)
        logits = model.decoder(z, max_length=SEQ_LEN)
        values = model.value_decoder(z)
        
        # Calculate losses
        loss_syntax_onehot, loss_syntax_consts, loss_value, loss = criterion(logits, values, y_rule_idx, y_consts, y_val)

        kl = model.kl(mu, sigma)
        alpha = anneal.alpha(step)
        elbo = loss + alpha*kl

        # Update parameters
        optimizer.zero_grad()
        elbo.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        assert not any(torch.isnan(param).any() for param in model.parameters()), "NaN found in model parameters"

        # Logging
        syntax_accuracy = calc_syntax_accuracy(logits, y_rule_idx)
        if step % PRINT_EVERY == 0:
            wandb.log({'train/loss': loss, 'train/loss_syntax_onehot': loss_syntax_onehot, 'train/loss_syntax_consts': loss_syntax_consts, 'train/loss_value': loss_value, 'train/kl': kl, 'train/alpha': alpha, 'train/elbo': elbo, 'train/syntax_accuracy': syntax_accuracy})

def test(test_loader):
    # Iterate over the training data

    loss_syntax_onehot_tot, loss_syntax_consts_tot, loss_value_tot, kl_tot, elbo_tot, loss_tot, syntax_accuracy_tot = 0, 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for step, (x, y_rule_idx, y_consts, y_val) in tqdm(enumerate(test_loader, 1), desc=f'Test', total=len(test_loader)):
            # Forward pass
            mu, sigma = model.encoder(x)
            z = model.sample(mu, sigma)
            logits = model.decoder(z, max_length=SEQ_LEN)
            values = model.value_decoder(z)
            
            # Calculate losses
            loss_syntax_onehot, loss_syntax_consts, loss_value, loss = criterion(logits, values, y_rule_idx, y_consts, y_val)

            kl = model.kl(mu, sigma)
            alpha = anneal.alpha(step)
            elbo = loss + alpha*kl

            syntax_accuracy = calc_syntax_accuracy(logits, y_rule_idx)

            loss_syntax_onehot_tot += loss_syntax_onehot
            loss_syntax_consts_tot += loss_syntax_consts
            loss_value_tot += loss_value
            kl_tot += kl
            elbo_tot += elbo
            loss_tot += loss
            syntax_accuracy_tot += syntax_accuracy

    loss_syntax_onehot_tot /= len(test_loader)
    loss_syntax_consts_tot /= len(test_loader)
    loss_value_tot /= len(test_loader)
    kl_tot /= len(test_loader)
    elbo_tot /= len(test_loader)
    loss_tot /= len(test_loader)
    syntax_accuracy_tot /= len(test_loader)

    wandb.log({'test/loss_syntax_onehot': loss_syntax_onehot_tot, 'test/loss_syntax_consts': loss_syntax_consts_tot, 'test/loss_value': loss_value_tot, 'test/kl': kl_tot, 'test/elbo': elbo_tot, 'test/loss': loss_tot, 'test/syntax_accuracy': syntax_accuracy_tot})


if __name__ == '__main__':
    # Init WandB
    run = wandb.init(
        project="similar-expressions-01",
        config={"learning_rate": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "value_loss_weight": VALUE_LOSS_WEIGHT, "const_loss_weight": CONST_LOSS_WEIGHT, "syntax_loss_weight": SYNTAX_LOSS_WEIGHT, "encoder_hidden": ENCODER_HIDDEN, "z_size": Z_SIZE, "decoder_hidden": DECODER_HIDDEN, "rnn_type": RNN_TYPE, "val_points": VAL_POINTS, "device": DEVICE}
    )

    # Init model
    torch.manual_seed(41)
    model = GrammarVAE(ENCODER_HIDDEN, Z_SIZE, DECODER_HIDDEN, TOKEN_CNT, RNN_TYPE, VAL_POINTS, DEVICE)

    def criterion(logits, values, y_rule_idx, y_consts, y_val):
        # TODO: Calc loss normalisation dynamically
        cross_entropy_prior_syntax = np.float32(1.3984073)
        mse_prior_consts = np.float32(0.06433585)
        mse_prior_values = np.float32(0.06500606)
        
        logits_onehot = logits[:, :, :-1]
        loss_syntax = torch.nn.CrossEntropyLoss()(logits_onehot.reshape(-1, logits_onehot.size(-1)), y_rule_idx.reshape(-1))/cross_entropy_prior_syntax
        loss_consts = torch.nn.MSELoss()(logits[:, :, -1], y_consts)/mse_prior_consts

        loss_value = torch.nn.MSELoss()(values, y_val)/mse_prior_values

        loss = loss_syntax*SYNTAX_LOSS_WEIGHT + loss_consts*CONST_LOSS_WEIGHT + loss_value*VALUE_LOSS_WEIGHT
        loss = loss / (SYNTAX_LOSS_WEIGHT + CONST_LOSS_WEIGHT + VALUE_LOSS_WEIGHT)

        return loss_syntax, loss_consts, loss_value, loss

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    anneal = AnnealKL(step=1e-3, rate=500)

    # Load data
    def value_transform(x):
        return torch.arcsinh(x)*0.1  # Example transformation. TODO: adjust scaling dynamically (arcsinh(1e5)=12.2 so currently this gives us 1.22)
    datapath = '/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data'
    train_loader, test_loader = create_dataloader(datapath, 'dataset_240817_2', test_split=0.1, batch_size=BATCH_SIZE, value_transform=value_transform, device=DEVICE)

    for epoch in range(1, EPOCHS+1):
        train_one_epoch(train_loader, epoch)
        test(test_loader)

    # torch.save(model, f'{wandb.run.dir}/model.pt')
    torch.save({'model_state_dict': model.state_dict()}, f'{wandb.run.dir}/model.pth')
