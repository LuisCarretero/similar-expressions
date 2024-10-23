import torch
from model import LitGVAE
import os
from config_util import load_config
from data_util import create_dataloader, calc_priors_and_means, summarize_dataloaders

def find_unused_parameters(model, data_loader):
    used_params = set()
    
    def hook(module, input, output):
        for param in module.parameters():
            used_params.add(param)
    
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook))
    

    for batch in data_loader:
        x0 = batch[0][0, ...].unsqueeze(0)  # Only get x and then only first element of batch
        break

    model(x0)
    
    # Remove the hooks
    for h in hooks:
        h.remove()
    
    unused_params = set(model.parameters()) - used_params
    return unused_params


def main(cfg_path, data_path, dataset_name):
    # Load config
    cfg_dict, cfg = load_config(cfg_path)

    # Load data
    train_loader, valid_loader, data_info = create_dataloader(data_path, dataset_name, cfg, num_workers=1)
    priors, _ = calc_priors_and_means(train_loader)  # TODO: Introduce bias initialization

    # In your main function or where you set up the model:
    gvae = LitGVAE(cfg, priors)
    unused = find_unused_parameters(gvae, train_loader)

    if unused:
        print("Unused parameters:")
        for param in unused:
            print(param.shape)
    else:
        print("All parameters are used.")

if __name__ == '__main__':
    cfg_path = 'src/model/config.json'  # /home/lc865/workspace/similar-expressions/src/model
    data_path = '/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/data'  # /Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data'  #  
    main(cfg_path, data_path, dataset_name='dataset_241008_1')  # dataset_240910_1, dataset_240822_1, dataset_240817_2