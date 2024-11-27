import torch.nn as ptn
from swak.pt.blocks import ActivatedBlock
from ...config import config

__all__ = [
    'vanilla_feedforward'
]

# ToDo: Make nice selection here so that only the needed one is instantiated!
vanilla_feedforward = ActivatedBlock(
    mod_dim=config.model.dim,
    activate=ptn.GELU(),
    drop=ptn.Dropout(config.model.dropout),
    hidden_factor=config.model.feedforward_factor,
    bias=config.model.bias,
    device=config.data.device,
    dtype=config.data.dtype
)
