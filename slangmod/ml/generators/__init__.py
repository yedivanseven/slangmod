from swak.funcflow import Curry
from ...config import config, Generators
from .abc import Generator
from .greedy import Greedy
from .random import Random
from .top_k import TopK
from .top_p import TopP
from .beamsearch import BeamSearch
from .styles import Style, style, space, paragraph, quote, dialogue

generator_type = {
    Generators.GREEDY: Greedy,
    Generators.RANDOM: Random,
    Generators.TOP_K: TopK,
    Generators.TOP_P: TopP,
    Generators.BEAM: BeamSearch
}[config.chat.generator]

create_generator = Curry(
    generator_type,
    style,
    temperature=config.chat.temperature,
    k=config.chat.k,
    p=config.chat.p,
    width=config.chat.width,
    penalty=config.chat.penalty
)

__all__ = [
    'Generator',
    'Greedy',
    'Random',
    'TopK',
    'TopP',
    'BeamSearch',
    'create_generator',
    'Style',
    'style',
    'space',
    'paragraph',
    'quote',
    'dialogue'
]
