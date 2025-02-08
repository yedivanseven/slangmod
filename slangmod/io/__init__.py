"""Tools for IO-related tasks like saving to and loading from disk.

User input and model output is handled by :obj:`~slangmod.io.clients` module.

Note
----
For more tools, see also the `pandas <https://yedivanseven.github.io/swak/pd.
html>`_ and `text <https://yedivanseven.github.io/swak/text.html>`_ sections
of the `swak <https://github.com/yedivanseven/swak>`_ documentation.

"""

from .clients import pre_trained_client
from .summary import save_config, save_train_toml
from .model import save_model, load_model
from .corpus import (
    NotFound,
    CorpusDiscovery,
    CorpusFilter,
    CorpusLoader,
    load_corpus,
    discover_raw,
    discover_corpus,
    discover_encodings,
    filter_train_files,
    filter_test_files,
    filter_validation_files
)
from .files import (
    DirectoryCleaner,
    clean_corpus_directory,
    clean_encodings_directory,
    write_clean_file,
    FileTypeExtractor,
    extract_file_type,
    extract_file_name,
    write_encoded_file,
    read_column
)
from .tokenizer import (
    TokenizerSaver,
    save_tokenizer,
    TokenizerLoader,
    load_tokenizer,
)

__all__ = [
    'pre_trained_client',
    'save_config',
    'save_train_toml',
    'save_model',
    'load_model',
    'NotFound',
    'CorpusDiscovery',
    'discover_raw',
    'discover_corpus',
    'discover_encodings',
    'CorpusFilter',
    'filter_train_files',
    'filter_test_files',
    'filter_validation_files',
    'CorpusLoader',
    'load_corpus',
    'DirectoryCleaner',
    'clean_corpus_directory',
    'clean_encodings_directory',
    'write_clean_file',
    'FileTypeExtractor',
    'extract_file_type',
    'extract_file_name',
    'write_encoded_file',
    'read_column',
    'TokenizerSaver',
    'save_tokenizer',
    'TokenizerLoader',
    'load_tokenizer'
]
