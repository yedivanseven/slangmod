from swak.funcflow import Curry
from ...config import config, Generators
from .abc import Generator
from .greedy import Greedy
from .random import Random
from .top_k import TopK
from .top_p import TopP
from .wrappers import Wrapper, wrapper, simple, paragraph, quote, dialogue

generator_type = {
    Generators.GREEDY: Greedy,
    Generators.RANDOM: Random,
    Generators.TOP_K: TopK,
    Generators.TOP_P: TopP,
}[config.chat.generator]

create_generator = Curry(
    generator_type,
    wrapper,
    k=config.chat.k,
    p=config.chat.p
)

__all__ = [
    'Generator',
    'Greedy',
    'Random',
    'TopK',
    'TopP',
    'create_generator',
    'Wrapper',
    'wrapper',
    'simple',
    'paragraph',
    'quote',
    'dialogue'
]
