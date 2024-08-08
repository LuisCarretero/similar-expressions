import os
import torch
from model import GrammarVAE
from util import AnnealKL, load_data, save_model
from grammar import GCFG
import wandb
from tqdm import tqdm

ENCODER_HIDDEN = 20
Z_SIZE = 3
DECODER_HIDDEN = 20
RNN_TYPE = 'lstm'
BATCH_SIZE = 32
MAX_LENGTH = 15
SYN_SEQ_LEN = len(GCFG.productions()) + 1
VAL_POINTS = 100
LR = 1e-2
CLIP = 5.
PRINT_EVERY = 200
EPOCHS = 5
VALUE_LOSS_WEIGHT = 0
CONST_LOSS_WEIGHT = 100


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
        logits = model.decoder(z, max_length=MAX_LENGTH)
        values = model.value_decoder(z)
        
        # Calculate losses
        loss_syntax_onehot, loss_syntax_consts = criterion_syntax(logits, y_rule_idx, y_consts)
        loss_value = criterion_value(values, y_val)
        loss = loss_syntax_onehot + loss_syntax_consts*CONST_LOSS_WEIGHT + loss_value*VALUE_LOSS_WEIGHT

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
            logits = model.decoder(z, max_length=MAX_LENGTH)
            values = model.value_decoder(z)
            
            # Calculate losses
            loss_syntax_onehot, loss_syntax_consts = criterion_syntax(logits, y_rule_idx, y_consts)
            loss_value = criterion_value(values, y_val)
            loss = loss_syntax_onehot + loss_syntax_consts*CONST_LOSS_WEIGHT + loss_value*VALUE_LOSS_WEIGHT

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
        config={"learning_rate": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "value_loss_weight": VALUE_LOSS_WEIGHT, "const_loss_weight": CONST_LOSS_WEIGHT, "encoder_hidden": ENCODER_HIDDEN, "z_size": Z_SIZE, "decoder_hidden": DECODER_HIDDEN, "rnn_type": RNN_TYPE, "val_points": VAL_POINTS}
    )

    # Init model
    torch.manual_seed(41)
    model = GrammarVAE(ENCODER_HIDDEN, Z_SIZE, DECODER_HIDDEN, SYN_SEQ_LEN, RNN_TYPE, VAL_POINTS, device='cpu')

    # Init loss funcitons
    criterion_syntax_onehot = torch.nn.CrossEntropyLoss()
    criterion_syntax_consts = torch.nn.MSELoss()
    def criterion_syntax(logits, y_rule_idx, y_consts):
        logits_onehot = logits[:, :, :-1]
        loss_syntax_onehot = criterion_syntax_onehot(logits_onehot.reshape(-1, logits_onehot.size(-1)), y_rule_idx.reshape(-1))
        loss_syntax_consts = criterion_syntax_consts(logits[:, :, -1], y_consts)
        return loss_syntax_onehot, loss_syntax_consts
    
    # criterion_value = lambda y_true, y_pred: torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100  # Dont punish absolute error
    criterion_value = torch.nn.L1Loss(reduction='mean')

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    anneal = AnnealKL(step=1e-3, rate=500)

    # Load data
    datapath = '/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data'
    train_loader, test_loader = load_data(datapath, 'expr_240807_5', test_split=0.1, batch_size=BATCH_SIZE)

    for epoch in range(1, EPOCHS+1):
        train_one_epoch(train_loader, epoch)
        test(test_loader)

    # save_model('model_240808_5_5epoch')
