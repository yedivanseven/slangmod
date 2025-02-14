import torch.nn as ptn
from swak.pt.misc import Identity, Compile
from swak.pt.blocks import (
    ActivatedHiddenBlock,
    GatedHiddenBlock,
    ActivatedGatedBlock
)
from ..config import config, Positions, Activations, Gates, FeedForwards
from .tokenizers import special
from .delayed import Delayed
from .models import (
    Sinusoidal,
    Learnable,
    Rotary,
    SelfAttention,
    EncoderLayer,
    Encoder
)

__all__ = [
    'create_model',
    'compile_model'
]

# Set up positional encodings according as requested in the config
emb_pos_enc = Delayed(
    {
        Positions.VANILLA: Sinusoidal,
        Positions.LEARNABLE: Learnable,
        Positions.SINUSOIDAL: Identity,
        Positions.ROTARY: Identity
    }[config.model.positions],
    mod_dim=config.model.dim,
    context=config.model.context,
    n_heads=config.model.n_heads,
    device=config.data.device,
    dtype=config.data.dtype
)
src_pos_enc = Delayed({
        Positions.VANILLA: Identity,
        Positions.LEARNABLE: Identity,
        Positions.SINUSOIDAL: Sinusoidal,
        Positions.ROTARY: Identity
    }[config.model.positions],
    mod_dim=config.model.dim,
    context=config.model.context,
    n_heads=config.model.n_heads,
    device=config.data.device,
    dtype=config.data.dtype
)
qk_pos_enc = Delayed({
        Positions.VANILLA: Identity,
        Positions.LEARNABLE: Identity,
        Positions.SINUSOIDAL: Identity,
        Positions.ROTARY: Rotary
    }[config.model.positions],
    mod_dim=config.model.dim,
    context=config.model.context,
    n_heads=config.model.n_heads,
    device=config.data.device,
    dtype=config.data.dtype
)

# Set up a self-attention according to specifications
self_attention = Delayed(
    SelfAttention,
    mod_dim=config.model.dim,
    n_heads=config.model.n_heads,
    bias=config.model.attn_bias,
    dropout=config.model.dropout,
    pos_enc=qk_pos_enc,
    device=config.data.device,
    dtype=config.data.dtype
)

# Pick an activation function for the feedforward component and, ...
activation = {
    Activations.ELU: ptn.ELU(),
    Activations.RELU: ptn.ReLU(),
    Activations.GELU: ptn.GELU(),
    Activations.SWISH: ptn.SiLU(),
    Activations.MISH: ptn.Mish(),
}[config.model.feedforward.activation]
# ... if needed, another one for gating.
gate = {
    Gates.SIGMOID: ptn.Sigmoid(),
    Gates.ELU: ptn.ELU(),
    Gates.RELU: ptn.ReLU(),
    Gates.GELU: ptn.GELU(),
    Gates.SWISH: ptn.SiLU(),
    Gates.MISH: ptn.Mish(),
    Gates.NONE: Identity()
}[config.model.feedforward.gate]

# Select the feedforward network specified in the config
feedforward = {
    FeedForwards.VANILLA: Delayed(
        ActivatedHiddenBlock,
        mod_dim=config.model.dim,
        activate=activation,
        drop=ptn.Dropout(config.model.dropout),
        hidden_factor=config.model.feedforward.factor,
        bias=config.model.feedforward.bias,
        device=config.data.device,
        dtype=config.data.dtype
    ),
    FeedForwards.GLU: Delayed(
        GatedHiddenBlock,
        mod_dim=config.model.dim,
        gate=gate,
        drop=ptn.Dropout(config.model.dropout),
        hidden_factor=config.model.feedforward.factor,
        bias=config.model.feedforward.bias,
        device=config.data.device,
        dtype=config.data.dtype
    ),
    FeedForwards.GRN: Delayed(
        ActivatedGatedBlock,
        mod_dim=config.model.dim,
        activate=activation,
        gate=gate,
        drop=ptn.Dropout(config.model.dropout),
        hidden_factor=config.model.feedforward.factor,
        bias=config.model.feedforward.bias,
        device=config.data.device,
        dtype=config.data.dtype
    )
}[config.model.feedforward.flavor]

# Configure an encoder layer with the components selected above
encoder_layer = Delayed(
    EncoderLayer,
    attention=self_attention,
    feed_forward=feedforward,
    pos_enc=src_pos_enc,
    bias=config.model.norm_bias,
    dropout=config.model.dropout,
    norm_cls=config.model.norm_cls,
    norm_first=config.model.norm_first,
    device=config.data.device,
    dtype=config.data.dtype
)

# Assemble the final model
create_model = Delayed(
    Encoder,
    vocab=config.tokens.vocab,
    layer=encoder_layer,
    n_layers=config.model.n_layers,
    pad_id=special.pad_id,
    pos_enc=emb_pos_enc,
    dropout=config.model.dropout,
    scale_grad_by_freq=config.model.scale_grad_by_freq,
    device=config.data.device,
    dtype=config.data.dtype
)

# Provide a ready-to-use compile for model training
compile_model = Compile(inplace=True, disable=config.model.disable)
