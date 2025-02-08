"""Various algorithms to produce model responses from user input."""

from swak.funcflow import Curry
from ...config import config, Generators
from .abc import Generator, NextToken
from .greedy import Greedy
from .top_k import TopK
from .top_p import TopP
from .beamsearch import BeamSearch

__all__ = [
    'Generator',
    'NextToken',
    'Greedy',
    'TopK',
    'TopP',
    'BeamSearch',
    'create_generator'
]

generator_type = {
    Generators.GREEDY: Greedy,
    Generators.TOP_K: TopK,
    Generators.TOP_P: TopP,
    Generators.BEAM: BeamSearch
}[config.chat.generator]

create_generator = Curry(
    generator_type,
    max_tokens=config.chat.max_tokens,
    temperature=config.chat.temperature,
    k=config.chat.k,
    p=config.chat.p,
    width=config.chat.width,
    boost=config.chat.boost
)
