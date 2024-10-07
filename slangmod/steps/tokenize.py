from tokenizers import Tokenizer
from swak.funcflow import Pipe, Partial
from ..io import save_tokenizer, books
from ..ml import train_tokenizer

tokenize = Pipe[tuple[()], [tuple[()]]](
    Partial[Tokenizer](
        train_tokenizer,
        books
    ),
    save_tokenizer
)
