import os
import torch
from model import GrammarVAE
from util import AnnealKL, load_raw_parsed_value_data, batch_iter
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
PRINT_EVERY = 100
EPOCHS = 1
VALUE_LOSS_WEIGHT = 0.4


def accuracy(logits, y):
    _, y_ = logits.max(-1)
    a = (y == y_).float().mean()
    return 100 * a.item()

def save_model(name: str):
    checkpoint_path = os.path.abspath('./smallMutations/similar-expressions/checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)

    while os.path.exists(f'{checkpoint_path}/{name}.pt'):
        name = f'{name}_copy'
    torch.save(model, f'{checkpoint_path}/{name}.pt')


def train(epoch: int):
    batch_cnt = (len(data_syntax) + BATCH_SIZE - 1) // BATCH_SIZE
    batches = batch_iter(data_syntax, data_value, BATCH_SIZE)
    for step, (x, y_rule_idx, y_consts, y_val) in tqdm(enumerate(batches, 1), total=batch_cnt, desc=f'Epoch {epoch}/{EPOCHS}'):

        mu, sigma = model.encoder(x)
        z = model.sample(mu, sigma)
        logits = model.decoder(z, max_length=MAX_LENGTH)
        values = model.value_decoder(z)
        
        loss_syntax = criterion_syntax(logits, y_rule_idx, y_consts)
        loss_value = criterion_value(values, y_val)
        loss = loss_syntax + loss_value * VALUE_LOSS_WEIGHT

        kl = model.kl(mu, sigma)

        alpha = anneal.alpha(step)
        elbo = loss + alpha*kl

        # Update parameters
        optimizer.zero_grad()
        elbo.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        if step % PRINT_EVERY == 0:
            wandb.log({'accuracy': -1, 'loss': loss, 'loss_syntax': loss_syntax, 'loss_value': loss_value, 'kl': kl, 'alpha': alpha, 'elbo': elbo})

if __name__ == '__main__':
    run = wandb.init(
    project="similar-expressions-01",
    config={
        "learning_rate": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "value_loss_weight": VALUE_LOSS_WEIGHT,
        "encoder_hidden": ENCODER_HIDDEN,
        "z_size": Z_SIZE,
        "decoder_hidden": DECODER_HIDDEN,
        "rnn_type": RNN_TYPE,
        "val_points": VAL_POINTS,
    },
)

    torch.manual_seed(41)

    model = GrammarVAE(ENCODER_HIDDEN, Z_SIZE, DECODER_HIDDEN, SYN_SEQ_LEN, RNN_TYPE, VAL_POINTS, device='cpu')

    criterion_syntax_onehot = torch.nn.CrossEntropyLoss()
    criterion_syntax_consts = torch.nn.MSELoss()
    def criterion_syntax(logits, y_rule_idx, y_consts):
        logits_onehot = logits[:, :, :-1]
        loss_syntax_onehot = criterion_syntax_onehot(logits_onehot.reshape(-1, logits_onehot.size(-1)), y_rule_idx.reshape(-1))
        loss_syntax_consts = criterion_syntax_consts(logits[:, :, -1], y_consts)
        return loss_syntax_onehot + loss_syntax_consts
    
    criterion_value = lambda y_true, y_pred: torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Load data
    datapath = '/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data'
    eqs, data_syntax, data_value = load_raw_parsed_value_data(datapath, 'expr_240807_5')
    data_syntax = torch.from_numpy(data_syntax).float().to(model.device)
    data_value = torch.from_numpy(data_value).float().to(model.device)

    anneal = AnnealKL(step=1e-3, rate=500)

    for epoch in range(1, EPOCHS+1):
        train()

    # save_model('model_240808_4_1epoch')
