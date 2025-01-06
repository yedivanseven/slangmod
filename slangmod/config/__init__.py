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
    FeedForward,
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
    'FeedForward',
    'Optimizers',
    'Scaling',
    'Generators',
    'Styles'
]

# Parse the environment for config options
parse_env = EnvParser()
env_vars = parse_env()
temporary = main(env_vars)

# Parse the command line for config options
parse_args = ArgParser(epilog=EPILOG.format(temporary))
actions, args = parse_args()
temporary = temporary(args)

# Load a config file if given
toml = {} if temporary.toml is None else TomlReader(temporary.toml)()
temporary = temporary(toml)

# Load the preset for the model size
load_preset = TextResourceLoader(__name__, 'presets')
preset = tomllib.loads(load_preset(temporary.preset))

# Update the original config in order of precedence
config = main(preset)(toml)(env_vars)(args)
