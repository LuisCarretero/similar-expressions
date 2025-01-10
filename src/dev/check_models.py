from src.model.util import load_config
from src.model.model import LitGVAE
from src.model.data_util import get_empty_priors, create_dataloader

from torch import randn
import torch


def single_forward():
    batch_size = 2  

    print('Loading config...')
    cfg_path = 'src/model/config.yaml'
    cfg = load_config(cfg_path)

    print('Initialising model...')
    gvae = LitGVAE(cfg, get_empty_priors())

    print('Running model and checking output shapes...')
    syntax_shape = (batch_size, cfg.model.io_format.seq_len, cfg.model.io_format.token_cnt)
    x = randn(syntax_shape)
    mean, ln_var = gvae.encoder(x)
    z = mean  # No sampling
    if cfg.training.mode != 'value_prediction':
        syntax_out = gvae.decoder(z)
        assert syntax_out.shape == syntax_shape, f'{syntax_out.shape = }'
    if cfg.training.mode != 'autoencoding':
        value_out = gvae.value_decoder(z)
        assert value_out.shape == (batch_size, cfg.model.io_format.val_points), f'{value_out.shape = }'

    print('All good!')


def single_forward_backward():
    batch_size = 2  

    print('Loading config...')
    cfg_path = 'src/model/config.yaml'
    cfg = load_config(cfg_path)

    print('Initialising model...')
    gvae = LitGVAE(cfg, get_empty_priors())

    print('Creating dataloader...')
    data_path = ['/mnt/cephfs/store/gr-mc2473/lc865/workspace/data', '/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/data'][0]
    dataset_name='dataset_241008_1'
    train_loader, valid_loader, data_info = create_dataloader(data_path, dataset_name, cfg, num_workers=1)

    print('Getting batch...')
    for batch in train_loader:
        x = batch[0]
        print(f'{x.shape = }')
        break

    print('Running forward pass...')
    x, y_syntax, y_consts, y_values = batch
    mean, ln_var, z, logits, values = gvae.forward(x)
    kl = gvae.calc_kl(mean, ln_var)  # Positive definite scalar, aim to minimize
    alpha = gvae.kl_anneal.alpha(gvae.current_epoch)
    loss, partial_losses = gvae.criterion(logits, values, y_syntax, y_consts, y_values, kl, alpha, z)

    print('Running backward pass...')
    loss.backward()
    
    print('Checking gradients...')
    for name, param in gvae.named_parameters():
        if param.grad is None:
            print(f'Warning: Found parameter with no gradient: {name}')
    
    print('Stepping optimizer...')
    optimizer = torch.optim.Adam(gvae.parameters(), lr=cfg.training.optimizer.lr)
    optimizer.step()
    print('All good!')

def check_dataloader():
    print('Loading config...')
    cfg_path = 'src/model/config.yaml'
    cfg = load_config(cfg_path)
    cfg.training.dataset_len_limit = 1000

    data_path = ['/mnt/cephfs/store/gr-mc2473/lc865/workspace/data', '/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/data'][0]
    dataset_name='dataset_241008_1'
    train_loader, valid_loader, data_info = create_dataloader(data_path, dataset_name, cfg, num_workers=1)

    for batch in train_loader:
        x = batch[0]
        print(f'{x.shape = }')
        break

if __name__ == '__main__':
    # check_dataloader()
    single_forward_backward()
