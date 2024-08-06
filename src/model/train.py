import os
import csv
import numpy as np
import torch
from model import GrammarVAE
from util import Timer, AnnealKL, load_onehot_data, batch_iter
from grammar import GCFG

ENCODER_HIDDEN = 20
Z_SIZE = 2
DECODER_HIDDEN = 20
RNN_TYPE = 'lstm'
BATCH_SIZE = 32
MAX_LENGTH = 15
OUTPUT_SIZE = len(GCFG.productions()) + 1
LR = 1e-2
CLIP = 5.
PRINT_EVERY = 100
EPOCHS = 1


def accuracy(logits, y):
    _, y_ = logits.max(-1)
    a = (y == y_).float().mean()
    return 100 * a.item()

def save(name):
    checkpoint_path = os.path.abspath('./smallMutations/similar-expressions/checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)

    while os.path.exists(f'{checkpoint_path}/{name}.pt'):
        name = f'{name}_copy'
    torch.save(model, f'{checkpoint_path}/{name}.pt')

def write_csv(d):
    log_path = os.path.abspath('./smallMutations/similar-expressions/log')
    os.makedirs(log_path, exist_ok=True)

    with open(f'{log_path}/log.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(d.keys())
        writer.writerows(zip(*d.values()))

def print_progress(step, log):
    print(
        '| step {}/{} | acc {:.2f} | loss {:.2f} | kl {:.2f} |'
        ' elbo {:.2f} | {:.0f} sents/sec |'.format(
            step, data.shape[0] // BATCH_SIZE,
            np.mean(log['acc'][-PRINT_EVERY:]),
            np.mean(log['loss'][-PRINT_EVERY:]),
            np.mean(log['kl'][-PRINT_EVERY:]),
            np.mean(log['elbo'][-PRINT_EVERY:]),
            BATCH_SIZE*PRINT_EVERY / timer.elapsed()
            )
        )
    write_csv(log)

def train():
    batches = batch_iter(data, BATCH_SIZE)
    for step, (x, y_rule_idx, y_consts) in enumerate(batches, 1):

        mu, sigma = model.encoder(x)
        z = model.sample(mu, sigma)
        logits = model.decoder(z, max_length=MAX_LENGTH)
        
        loss = criterion(logits, y_rule_idx, y_consts)
        kl = model.kl(mu, sigma)

        alpha = anneal.alpha(step)
        elbo = loss + alpha*kl

        # Update parameters
        optimizer.zero_grad()
        elbo.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        log['loss'].append(loss.item())
        log['kl'].append(kl.item())
        log['elbo'].append(elbo.item())
        log['acc'].append(-1)  # accuracy(logits, y)

        if step % PRINT_EVERY == 0:
            print_progress(step, log)
            write_csv(log)


if __name__ == '__main__':
    torch.manual_seed(41)

    model = GrammarVAE(ENCODER_HIDDEN, Z_SIZE, DECODER_HIDDEN, OUTPUT_SIZE, RNN_TYPE, device='cpu')

    criterion_onehot = torch.nn.CrossEntropyLoss()
    criterion_consts = torch.nn.MSELoss()
    def criterion(logits, y_rule_idx, y_consts):
        logits_onehot = logits[:, :, :-1]
        loss_onehot = criterion_onehot(logits_onehot.reshape(-1, logits_onehot.size(-1)), y_rule_idx.reshape(-1))
        loss_consts = criterion_consts(logits[:, :, -1], y_consts)
        return loss_onehot + loss_consts

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Load data
    parsed_path = r'/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data/onehot_parsed.h5'
    data = load_onehot_data(parsed_path)
    data = torch.from_numpy(data).float().to(model.device)  # Turn it into a float32 PyTorch Tensor
    data.clamp_(-1e2, 1e2) # Quickfix: Some constants are too large, causing NaNs in the decoder

    timer = Timer()
    log = {'loss': [], 'kl': [], 'elbo': [], 'acc': []}
    anneal = AnnealKL(step=1e-3, rate=500)

    try:
        for epoch in range(1, EPOCHS+1):
            print('-' * 69 + '\nEpoch {}/{}\n'.format(epoch, EPOCHS) + '-' * 69 + '\n')
            train()
    except KeyboardInterrupt:
        print('-' * 69 + '\nExiting training early\n' + '-' * 69 + '\n')

    save('model_const_clamped')
    write_csv(log)
