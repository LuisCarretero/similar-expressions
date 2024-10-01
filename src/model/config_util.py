import json
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, List

@dataclass
class EncoderConfig:
    z_slice: List[int]
    size_hidden: int
    conv_size: Literal["small", "large", "extra_large"]

@dataclass
class DecoderConfig:
    z_slice: List[int]
    size_hidden: int
    rnn_type: Literal["lstm", "gru"]

@dataclass
class ValueDecoderConfig:
    conv_size: Literal["small", "medium", "large"]
    
@dataclass
class IoFormatConfig:
    seq_len: int
    token_cnt: int
    val_points: int

@dataclass
class ModelConfig:
    encoder: EncoderConfig
    z_size: int
    decoder: DecoderConfig
    value_decoder: ValueDecoderConfig
    io_format: IoFormatConfig

@dataclass
class CriterionConfig:
    ae_weight: float
    kl_weight: float
    syntax_weight: float

@dataclass
class SamplingConfig:
    prior_std: float
    eps: float

@dataclass
class OptimizerConfig:
    lr: float
    clip: float
    scheduler_factor: float
    scheduler_patience: int
    scheduler_threshold: float
    scheduler_min_lr: float

@dataclass
class AnnealConfig:
    schedule: Literal["sigmoid"]
    midpoint: float
    steepness: float

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    valid_split: float
    dataset_len_limit: Optional[int]
    criterion: CriterionConfig
    sampling: SamplingConfig
    optimizer: OptimizerConfig
    kl_anneal: AnnealConfig
    values_init_bias: bool
    use_grammar_mask: bool

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig

def load_config(file_path: str) -> Tuple[dict, Config]:
    with open(file_path, 'r') as f:
        cfg_dict = json.load(f)

    return cfg_dict, dict_to_config(cfg_dict)

def dict_to_config(cfg_dict: dict, fallback_dict: dict = None) -> Config:
    # Define fallback values
    if not fallback_dict:
        fallback_dict = {
            'model': {
                'encoder': {
                    'size_hidden': 128,
                    'conv_size': 'large'
                },
                'z_size': 128,
                'decoder': {
                    'z_slice': [0, -1],
                    'size_hidden': 64,
                    'rnn_type': 'lstm'
                },
                'value_decoder': {
                    'z_slice': [0, -1],
                    'size_lin1': 64,
                    'conv_size': 'medium'  # FIXME: Rename
                },
                'io_format': {
                    'seq_len': 15,
                    'token_cnt': 10,
                    'val_points': 100
                }
            },
            'training': {
                'batch_size': 128,
                'epochs': 4,
                'valid_split': 0.1,
                'dataset_len_limit': None,
                'criterion': {
                    'ae_weight': 1,
                    'kl_weight': 1,
                    'syntax_weight': 0.5
                },
                'sampling': {
                    'prior_std': 0.1,
                    'eps': 1
                },
                'optimizer': {
                    'lr': 2e-3,
                    'clip': 5.0,
                    'scheduler_factor': 0.1,
                    'scheduler_patience': 10,
                    'scheduler_threshold': 1e-4,
                    'scheduler_min_lr': 1e-6
                },
                'kl_anneal': {
                    'schedule': 'sigmoid',
                    'midpoint': 0.3,
                    'steepness': 10
                },
                'values_init_bias': False,
                'use_grammar_mask': False
            }
        }

    # Merge fallback values with provided cfg_dict and print messages for fallback usage
    merged_cfg = {}
    for section in ['model', 'training']:
        merged_cfg[section] = {}
        for key, default_value in fallback_dict[section].items():
            if section not in cfg_dict or key not in cfg_dict[section]:
                print(f"Using fallback value for {section}.{key}: {default_value}")
                merged_cfg[section][key] = default_value
            else:
                if isinstance(default_value, dict):
                    merged_cfg[section][key] = {}
                    for subkey, subdefault in default_value.items():
                        if subkey not in cfg_dict[section][key]:
                            print(f"Using fallback value for {section}.{key}.{subkey}: {subdefault}")
                            merged_cfg[section][key][subkey] = subdefault
                        else:
                            merged_cfg[section][key][subkey] = cfg_dict[section][key][subkey]
                else:
                    merged_cfg[section][key] = cfg_dict[section][key]

    def create_config_with_error_check(config_class, config_dict):
        expected_keys = set(config_class.__annotations__.keys())
        provided_keys = set(config_dict.keys())
        unexpected_keys = provided_keys - expected_keys
        if unexpected_keys:
            for key in unexpected_keys:
                print(f"Unexpected key in {config_class.__name__}: {key}")
        return config_class(**{k: v for k, v in config_dict.items() if k in expected_keys})
    

    print(merged_cfg['model']['value_decoder'])
    return Config(
        model=ModelConfig(
            encoder=create_config_with_error_check(EncoderConfig, merged_cfg['model']['encoder']),
            z_size=merged_cfg['model']['z_size'],
            decoder=create_config_with_error_check(DecoderConfig, merged_cfg['model']['decoder']),
            value_decoder=create_config_with_error_check(ValueDecoderConfig, merged_cfg['model']['value_decoder']),
            io_format=create_config_with_error_check(IoFormatConfig, merged_cfg['model']['io_format'])
        ),
        training=TrainingConfig(
            batch_size=merged_cfg['training']['batch_size'],
            epochs=merged_cfg['training']['epochs'],
            valid_split=merged_cfg['training']['valid_split'],
            dataset_len_limit=merged_cfg['training']['dataset_len_limit'],
            criterion=create_config_with_error_check(CriterionConfig, merged_cfg['training']['criterion']),
            sampling=create_config_with_error_check(SamplingConfig, merged_cfg['training']['sampling']),
            optimizer=create_config_with_error_check(OptimizerConfig, merged_cfg['training']['optimizer']),
            kl_anneal=create_config_with_error_check(AnnealConfig, merged_cfg['training']['kl_anneal']),
            values_init_bias=merged_cfg['training']['values_init_bias'],
            use_grammar_mask=merged_cfg['training']['use_grammar_mask']
        )
    )
