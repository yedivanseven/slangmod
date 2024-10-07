from pathlib import Path
from swak.misc import ArgRepr
from ..config import config


class BookLoader(ArgRepr):

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.path = str(Path(config.books).resolve())

    def __str__(self) -> str:
        return '\n'.join([f'"{book}"' for book in self.books])

    @property
    def books(self) -> list[str]:
        return [
            str(item.resolve())
            for item in Path(self.path).iterdir()
            if item.is_file() and item.suffix == '.txt'
        ]

    @staticmethod
    def read(book: str) -> str:
        with Path(book).open() as stream:
            text = stream.read()
        return text

    def __call__(self) -> str:
        return '\n\n'.join(map(self.read, self.books))


load_books = BookLoader(config.books)
