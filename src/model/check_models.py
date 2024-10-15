from config_util import load_config
from model import LitGVAE
from data_util import get_empty_priors, create_dataloader
import torch


def main():
    print('Loading config...')
    cfg_path = 'src/model/config.json'
    _, cfg = load_config(cfg_path)

    print('Initialising model...')
    gvae = LitGVAE(cfg, get_empty_priors())

    print('Checking model...')
    syntax_shape = (1, cfg.model.io_format.seq_len, cfg.model.io_format.token_cnt)
    x = torch.randn(syntax_shape)
    mu, sigma = gvae.encoder(x)
    z = mu  # No sampling
    syntax_out = gvae.decoder(z)
    value_out = gvae.value_decoder(z)

    print('Checking output shapes...')
    assert syntax_out.shape == syntax_shape, f'{syntax_out.shape = }'
    assert value_out.shape == (1, cfg.model.io_format.val_points), f'{value_out.shape = }'
    print('All good!')

def check_dataloader():
    print('Loading config...')
    cfg_path = 'src/model/config.json'
    _, cfg = load_config(cfg_path)

    data_path = ['/store/DAMTP/lc865/workspace/data', '/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/data'][0]
    dataset_name='dataset_241008_1'
    train_loader, valid_loader, data_info = create_dataloader(data_path, dataset_name, cfg, num_workers=1)

    for batch in train_loader:
        x = batch[0]
        print(f'{x.shape = }')
        break


if __name__ == '__main__':
    # check_dataloader()
    main()
