from typing import Any
import torch.nn as ptn
from swak.pt.types import Module
from swak.pt.misc import Identity, Compile
from swak.pt.blocks import (
    ActivatedHiddenBlock,
    GatedHiddenBlock,
    ActivatedGatedBlock
)
from ..config import config, Positions, Activations, Gates, FeedForwards
from .tokenizers import special
from .models import (
    Sinusoidal,
    Learnable,
    Rotary,
    SelfAttention,
    EncoderLayer,
    Encoder,
    Reference
)

__all__ = [
    'create_model',
    'compile_model'
]


# ToDo: Make a "Delayed" class instead"!
def create_model(*_: Any, **__: Any) -> Module:
    """To delay allocating CPU/GPU memory until actually needed."""
    # Chose positional encodings according to flavor
    emb_pos_enc = {
        Positions.VANILLA: Sinusoidal,
        Positions.LEARNABLE: Learnable,
        Positions.SINUSOIDAL: Identity,
        Positions.ROTARY: Identity
    }[config.model.positions](
        mod_dim=config.model.dim,
        context=config.model.context,
        n_heads=config.model.n_heads,
        device=config.data.device,
        dtype=config.data.dtype
    )
    src_pos_enc = {
        Positions.VANILLA: Identity,
        Positions.LEARNABLE: Identity,
        Positions.SINUSOIDAL: Sinusoidal,
        Positions.ROTARY: Identity
    }[config.model.positions](
        mod_dim=config.model.dim,
        context=config.model.context,
        n_heads=config.model.n_heads,
        device=config.data.device,
        dtype=config.data.dtype
    )
    qk_pos_enc = {
        Positions.VANILLA: Identity,
        Positions.LEARNABLE: Identity,
        Positions.SINUSOIDAL: Identity,
        Positions.ROTARY: Rotary
    }[config.model.positions](
        mod_dim=config.model.dim,
        context=config.model.context,
        n_heads=config.model.n_heads,
        device=config.data.device,
        dtype=config.data.dtype
    )
    # Instantiate a self-attention if needed
    self_attention = Identity() if config.model.reference else SelfAttention(
        mod_dim=config.model.dim,
        n_heads=config.model.n_heads,
        bias=config.model.bias,
        dropout=config.model.dropout,
        pos_enc=qk_pos_enc,
        device=config.data.device,
        dtype=config.data.dtype
    )
    # Pick an activation function for the feedforward layer and ...
    activation = {
        Activations.ELU: ptn.ELU(),
        Activations.RELU: ptn.ReLU(),
        Activations.GELU: ptn.GELU(),
        Activations.SWISH: ptn.SiLU(),
        Activations.MISH: ptn.Mish(),
    }[config.model.feedforward.activation]
    # ... another one for the gating, if needed.
    gate = {
        Gates.SIGMOID: ptn.Sigmoid(),
        Gates.ELU: ptn.ELU(),
        Gates.RELU: ptn.ReLU(),
        Gates.GELU: ptn.GELU(),
        Gates.SWISH: ptn.SiLU(),
        Gates.MISH: ptn.Mish(),
        Gates.NONE: Identity()
    }[config.model.feedforward.gate]
    # Instantiate a feedforward network with the chosen activation/gate
    # ToDo: Move back to feedforward!
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
    # Create an encoder layer according to settings
    encoder_layer = Identity() if config.model.reference else EncoderLayer(
        attention=self_attention,
        feed_forward=feedforward,
        pos_enc=src_pos_enc,
        bias=config.model.bias,
        dropout=config.model.dropout,
        norm_first=config.model.norm_first,
        device=config.data.device,
        dtype=config.data.dtype
    )
    # Finally, instantiate the actual model as specified in the config
    return Reference(
        mod_dim=config.model.dim,
        vocab=config.tokens.vocab,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        pos_enc=Sinusoidal(
            mod_dim=config.model.dim,
            context=config.model.context,
            device=config.data.device,
            dtype=config.data.dtype
        ),
        feedforward_factor=config.model.feedforward.factor,
        scale_grad_by_freq=config.model.scale_grad_by_freq,
        dropout=config.model.dropout,
        bias=config.model.bias,
        norm_first=config.model.norm_first,
        device=config.data.device,
        dtype=config.data.dtype
    ) if config.model.reference else Encoder(
        vocab=config.tokens.vocab,
        layer=encoder_layer,
        n_layers=config.model.n_layers,
        pad_id=special.pad_id,
        pos_enc=emb_pos_enc,
        bias=config.model.bias,
        dropout=config.model.dropout,
        scale_grad_by_freq=config.model.scale_grad_by_freq,
        device=config.data.device,
        dtype=config.data.dtype
    )

# ToDo: Make sure in-place works when re-loading model!
compile_model = Compile(inplace=True, disable=config.model.disable_compile)
