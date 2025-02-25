from typing import Dict
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


def load_config(file_path: str, use_fallback: bool = False, fallback_cfg_path: str = 'config.yaml') -> DictConfig:
    # TODO: Add error handling, fallback values, etc.
    cfg = OmegaConf.load(file_path)

    if use_fallback:
        fallback_cfg = OmegaConf.load(fallback_cfg_path)
        cfg = OmegaConf.merge(fallback_cfg, cfg)  # second arg overrides first

    return cfg

def update_cfg(default_cfg: DictConfig, partial_cfg: Dict):
    """
    Update a default config with a partial config. Used for WandB sweeps.
    """
    for k, v in partial_cfg.items():
        tmp_cfg = default_cfg
        path = k.split('.')
        for dir in path[:-1]:
            tmp_cfg = tmp_cfg[dir]
        tmp_cfg[path[-1]] = v

def dict_lit2num(d: Dict, verbose=False):
    """
    Convert all literal values in a dictionary to numbers (int or float). Needed as WandB stores all values as strings?
    """
    def _convert(x):
        if isinstance(x, dict):
            return {k: _convert(v) for k, v in x.items()}
        else:  # Leaf node
            try:
                tmp = float(x)
                if tmp.is_integer():
                    tmp = int(tmp)
            except:
                tmp = x
            if verbose:
                print(f'Leaf node:{x} -> {tmp}; type: {type(tmp)}')
            return tmp
    return _convert(d)
