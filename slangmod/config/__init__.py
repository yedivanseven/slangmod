import tomllib
import warnings
from swak.cli import EnvParser, ArgParser, EPILOG
from swak.text import TextResourceLoader, TomlReader
from .defaults import main, Main
from .enums import (
    Devices,
    LiteralDevice,
    Dtypes,
    Cleaners,
    Tokenizers,
    Positions,
    Norms,
    LiteralNorm,
    Activations,
    Gates,
    FeedForwards,
    Optimizers,
    Scaling,
    Generators,
    Styles
)

__all__ = [
    'Main',
    'config',
    'actions',
    'Devices',
    'LiteralDevice',
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

# Parse the environment for config options
parse_env = EnvParser(prefix='SLM_')
env_vars = parse_env()
temporary = main(env_vars)

# Parse the command line for config options
ACTIONS = """actions:
dry-run       Print the configuration slangmod would run with.
clean         Clean wiki40b-en and gutenberg data.
tokenize      Train a tokenizer on the cleaned corpus.
encode        Encode text documents with the trained tokenizer.
train         Train the specified model on the encoded data.
monitor       Print a gnuplot script to graphically track training progress.
compare       Print a gnuplot script to compare convergence of training runs.
summarize     Print a JSON of all training-run summaries in one experiment.
chat          Start a console client to chat with a trained model.
"""
parse_args = ArgParser(description=ACTIONS, epilog=EPILOG.format(temporary))
actions, args = parse_args()
temporary = temporary(args)

# If no explict config file is given, ...
if temporary.toml is None:
    # ... try to load the default ...
    try:
        toml = TomlReader(temporary.default)()
    # ... and warn if that's not found.
    except FileNotFoundError:
        cached_formatter = warnings.formatwarning
        warnings.formatwarning = lambda message, *_: str(message)
        msg = 'Default config file {} not found.\nProceeding with defaults.\n'
        warnings.warn(msg.format(temporary.default))
        warnings.formatwarning = cached_formatter
        toml = {}
# ... but if it is, then just read it.
else:
    toml = TomlReader(temporary.toml)()
temporary = temporary(toml)

# Load the preset for the model size
load_preset = TextResourceLoader(__name__, 'presets')
preset = tomllib.loads(load_preset(temporary.preset))

# Update the original config in order of precedence
config = main(preset)(toml)(env_vars)(args, _actions=actions)

# Remove the "resume" action if given, since it only modifies other actions
actions = [action for action in actions if action.strip().lower() != 'resume']
