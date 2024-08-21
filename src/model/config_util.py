import json
from dataclasses import dataclass
from typing import Literal, Tuple

@dataclass
class EncoderConfig:
    size_hidden: int
    conv_size: Literal["large"]

@dataclass
class DecoderConfig:
    size_hidden: int
    rnn_type: Literal["lstm"]

@dataclass
class ValueDecoderConfig:
    size_lin1: int

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
    values_loss_weight: float
    consts_loss_weight: float
    syntax_loss_weight: float

@dataclass
class OptimizerConfig:
    lr: float
    clip: float

@dataclass
class TrainingConfig:
    batch_size: int
    print_every: int
    epochs: int
    criterion: CriterionConfig
    optimizer: OptimizerConfig

@dataclass
class MiscConfig:
    device: Literal["cpu"]
    values_init_bias: bool

@dataclass
class Config:
    architecture: ModelConfig
    training: TrainingConfig
    misc: MiscConfig

def load_config(file_path: str) -> Tuple[dict, Config]:
    with open(file_path, 'r') as f:
        cfg_dict = json.load(f)

    return cfg_dict, dict_to_config(cfg_dict)

def dict_to_config(cfg_dict: dict) -> Config:
    return Config(
        architecture=ModelConfig(
            encoder=EncoderConfig(**cfg_dict['architecture']['encoder']),
            z_size=cfg_dict['architecture']['z_size'],
            decoder=DecoderConfig(**cfg_dict['architecture']['decoder']),
            value_decoder=ValueDecoderConfig(**cfg_dict['architecture']['value_decoder']),
            io_format=IoFormatConfig(**cfg_dict['architecture']['io_format'])
        ),
        training=TrainingConfig(
            batch_size=cfg_dict['training']['batch_size'],
            print_every=cfg_dict['training']['print_every'],
            epochs=cfg_dict['training']['epochs'],
            criterion=CriterionConfig(**cfg_dict['training']['criterion']),
            optimizer=OptimizerConfig(**cfg_dict['training']['optimizer'])
        ),
        misc=MiscConfig(**cfg_dict['misc'])
    )

# Usage example:
# config = load_config('similar-expressions/src/model/hyperparameters.json')