import torch.nn as ptn
from swak.pt.blocks import ActivatedBlock
from ...config import config

__all__ = [
    'vanilla_feedforward'
]

activation = ptn.Sequential(ptn.GELU(), ptn.Dropout(config.model.dropout))
vanilla_feedforward = ActivatedBlock(
    mod_dim=config.model.dim,
    activate=activation,
    hidden_factor=config.model.feedforward_factor,
    bias=config.model.bias,
    device=config.data.device,
    dtype=config.data.dtype
)
