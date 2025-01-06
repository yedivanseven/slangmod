import torch.nn as ptn
from swak.pt.misc import Identity
from swak.pt.blocks import (
    ActivatedHiddenBlock,
    GatedHiddenBlock,
    ActivatedGatedBlock
)
from ...config import config, FeedForward, Activations

__all__ = [
    'feedforward',
]

activation = {
    Activations.ELU: ptn.ELU(),
    Activations.RELU: ptn.ReLU(),
    Activations.GELU: ptn.GELU(),
    Activations.SWISH: ptn.SiLU(),
    Activations.MISH: ptn.Mish(),
}[config.model.feedforward.activation]

if config.model.reference:
    feedforward = Identity()
elif config.model.feedforward.flavour == FeedForward.VANILLA:
    feedforward = ActivatedHiddenBlock(
        mod_dim=config.model.dim,
        activate=activation,
        drop=ptn.Dropout(config.model.dropout),
        hidden_factor=config.model.feedforward.factor,
        bias=config.model.bias,
        device=config.data.device,
        dtype=config.data.dtype
    )
elif config.model.feedforward.flavour == FeedForward.GLU:
    feedforward = GatedHiddenBlock(
        mod_dim=config.model.dim,
        gate=activation,
        drop=ptn.Dropout(config.model.dropout),
        hidden_factor=config.model.feedforward.factor,
        bias=config.model.bias,
        device=config.data.device,
        dtype=config.data.dtype
    )
elif config.model.feedforward.flavour == FeedForward.GRN:
    feedforward = ActivatedGatedBlock(
        mod_dim=config.model.dim,
        activate=activation,
        gate=activation,
        drop=ptn.Dropout(config.model.dropout),
        hidden_factor=config.model.feedforward.factor,
        bias=config.model.bias,
        device=config.data.device,
        dtype=config.data.dtype
    )
else:
    msg = 'Feed-forward flavour {} is not implemented!'
    raise NotImplementedError(msg.format(config.model.feedforward.flavour))
