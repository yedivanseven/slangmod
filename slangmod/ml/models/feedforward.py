import torch.nn as ptn
from swak.pt.blocks import ActivatedBlock
from ...config import config

__all__ = [
    'vanilla_feedforward'
]

vanilla_activation = ptn.Sequential(
    ptn.GELU(),
    ptn.Dropout(config.model.dropout)
)
# ToDo: Make nice selection here so that only the needed one is instantiated!
vanilla_feedforward = ActivatedBlock(
    mod_dim=config.model.dim,
    activate=vanilla_activation,
    hidden_factor=config.model.feedforward_factor,
    bias=config.model.bias,
    device=config.data.device,
    dtype=config.data.dtype
)
