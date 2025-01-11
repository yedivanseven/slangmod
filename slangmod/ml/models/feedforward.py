import torch.nn as ptn
from swak.pt.misc import Identity
from swak.pt.blocks import (
    ActivatedHiddenBlock,
    GatedHiddenBlock,
    ActivatedGatedBlock
)
from ...config import config, FeedForwards, Activations, Gates

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

gate = {
    Gates.SIGMOID: ptn.Sigmoid(),
    Gates.ELU: ptn.ELU(),
    Gates.RELU: ptn.ReLU(),
    Gates.GELU: ptn.GELU(),
    Gates.SWISH: ptn.SiLU(),
    Gates.MISH: ptn.Mish(),
    Gates.NONE: Identity()
}[config.model.feedforward.gate]

if config.model.reference:
    feedforward = Identity()
elif config.model.feedforward.flavour == FeedForwards.VANILLA:
    feedforward = ActivatedHiddenBlock(
        mod_dim=config.model.dim,
        activate=activation,
        drop=ptn.Dropout(config.model.dropout),
        hidden_factor=config.model.feedforward.factor,
        bias=config.model.bias,
        device=config.data.device,
        dtype=config.data.dtype
    )
elif config.model.feedforward.flavour == FeedForwards.GLU:
    feedforward = GatedHiddenBlock(
        mod_dim=config.model.dim,
        gate=gate,
        drop=ptn.Dropout(config.model.dropout),
        hidden_factor=config.model.feedforward.factor,
        bias=config.model.bias,
        device=config.data.device,
        dtype=config.data.dtype
    )
elif config.model.feedforward.flavour == FeedForwards.GRN:
    feedforward = ActivatedGatedBlock(
        mod_dim=config.model.dim,
        activate=activation,
        gate=gate,
        drop=ptn.Dropout(config.model.dropout),
        hidden_factor=config.model.feedforward.factor,
        bias=config.model.bias,
        device=config.data.device,
        dtype=config.data.dtype
    )
else:
    msg = 'Feed-forward flavour {} is not implemented!'
    raise NotImplementedError(msg.format(config.model.feedforward.flavour))
