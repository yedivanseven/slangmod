import tomllib
from swak.cli import EnvParser, ArgParser, EPILOG
from swak.text import TextResourceLoader, TomlReader
from .defaults import main, Main
from .enums import (
    Devices,
    LiteralDevice,
    Tokenizers,
    Positions,
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
    'Tokenizers',
    'Positions',
    'Activations',
    'Gates',
    'FeedForwards',
    'Optimizers',
    'Scaling',
    'Generators',
    'Styles'
]
# ToDo: Rename presets 1x1_32.toml etc ...
# Parse the environment for config options
parse_env = EnvParser()
env_vars = parse_env()
temporary = main(env_vars)

ACTIONS = """actions:
dry-run       Print the configuration slangmod would run with.
tokenizer     Train a tokenizer on the corpus.
encode        Encode text documents with the trained tokenizer.
train         Train the specified model ont hne encoded data.
chat          Start a console client to chat with a trained model.
"""
# Parse the command line for config options
parse_args = ArgParser(description=ACTIONS, epilog=EPILOG.format(temporary))
actions, args = parse_args()
temporary = temporary(args)

# Load a config file if given
toml = {} if temporary.toml is None else TomlReader(temporary.toml)()
temporary = temporary(toml)

# Load the preset for the model size
load_preset = TextResourceLoader(__name__, 'presets')
preset = tomllib.loads(load_preset(temporary.preset))

# Update the original config in order of precedence
config = main(preset)(toml)(env_vars)(args, actions=actions)

# Remove the "resume" action if given, since it only modifies others
actions = [action for action in actions if action != 'resume']
