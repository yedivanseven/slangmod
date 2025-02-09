from enum import StrEnum
from typing import Literal

__all__ = [
    'LiteralDevice',
    'Devices',
    'Dtypes',
    'Cleaners',
    'Tokenizers',
    'Positions',
    'Norms',
    'LiteralNorm',
    'Activations',
    'Gates',
    'FeedForwards',
    'Optimizers',
    'Scaling',
    'Generators',
    'Styles'
]

type LiteralDevice = Literal['cpu', 'cuda']
type LiteralNorm = Literal['layer', 'rms']


class Devices(StrEnum):
    """Device to run model training on."""
    CPU = 'cpu'
    CUDA = 'cuda'


class Dtypes(StrEnum):
    """Permissible torch dtypes."""
    FLOAT = 'float'
    FLOAT32 = 'float32'
    BFLOAT = 'bfloat16'
    BFLOAT16 = 'bfloat16'


class Cleaners(StrEnum):
    """Data cleaning steps to perform."""
    QUOTES = 'quotes'
    ENCODING = 'encoding'
    WIKI40B = 'wiki40b'


class Tokenizers(StrEnum):
    """Tokenization algorithm to use."""
    BPE = 'bpe'
    WORDPIECE = 'wordpiece'
    UNIGRAM = 'unigram'


class Positions(StrEnum):
    """Which type of positional encoding to use and where."""
    VANILLA = 'vanilla'
    LEARNABLE = 'learnable'
    SINUSOIDAL = 'sinusoidal'
    ROTARY = 'rotary'

class Norms(StrEnum):
    """The type of norm used between (sub-)layers."""
    LAYER = 'layer'
    RMS = 'rms'


class Activations(StrEnum):
    """Activation functions to use in the feed-forward part of the model."""
    ELU = 'elu'
    RELU = 'relu'
    GELU = 'gelu'
    SWISH = 'swish'
    MISH = 'mish'


class Gates(StrEnum):
    """Non-linearity to use in gated linear units, if present."""
    SIGMOID = 'sigmoid'
    ELU = 'elu'
    RELU = 'relu'
    GELU = 'gelu'
    SWISH = 'swish'
    MISH = 'mish'
    NONE = 'none'


class FeedForwards(StrEnum):
    """Which type of feed-forward network to use."""
    VANILLA = 'vanilla'
    GLU = 'glu'
    GRN = 'grn'


class Optimizers(StrEnum):
    """Which optimizer to use for model training."""
    ADMAW = 'adamw'
    ADAFACTOR = 'adafactor'


class Scaling(StrEnum):
    """How to scale down the learning rate over time."""
    INVERSE = 'inverse'
    EXPONENTIAL = 'exponential'
    COSINE = 'cosine'


class Generators(StrEnum):
    """How to generate model responses from next-token prediction."""
    GREEDY = 'greedy'
    TOP_K = 'top_k'
    TOP_P = 'top_p'
    BEAM = 'beam'


class Styles(StrEnum):
    """Formatting options for user input and model response."""
    SPACE = 'space'
    PARAGRAPH = 'paragraph'
    QUOTE = 'quote'
    DIALOGUE = 'dialogue'
