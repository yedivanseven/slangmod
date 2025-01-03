import torch.nn as ptn
from swak.pt.misc import Identity
from swak.pt.blocks import ActivatedBlock
from ...config import config, FeedForward

__all__ = [
    'feedforward'
]

if config.model.reference:
    feedforward = Identity()
elif config.model.feedforward_flavour == FeedForward.VANILLA:
    feedforward = ActivatedBlock(
        mod_dim=config.model.dim,
        activate=ptn.GELU(),
        drop=ptn.Dropout(config.model.dropout),
        hidden_factor=config.model.feedforward_factor,
        bias=config.model.bias,
        device=config.data.device,
        dtype=config.data.dtype
    )
else:
    msg = 'Feed-forward flavour {} is not implemented!'
    raise NotImplementedError(msg.format(config.model.feedforward_flavour))
