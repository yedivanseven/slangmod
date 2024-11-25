import torch.nn.functional as ptnf
from swak.pt.blocks import ActivatedBlock
from ...config import config

__all__ = [
    'vanilla_feedforward'
]

vanilla_feedforward = ActivatedBlock(
    mod_dim=config.model.dim,
    activate=ptnf.gelu,
    hidden_factor=config.config.model.feedforward_factor,
    bias=config.model.bias,
    device=config.data.device,
    dtype=config.data.dtype
)
