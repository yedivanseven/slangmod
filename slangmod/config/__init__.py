from swak.cli import EnvParser, ArgParser, EPILOG
from .defaults import main, Main

__all__ = [
    'Main',
    'config',
    'actions',
]

# Parse the environment and update the default config accordingly
parse_env = EnvParser()
env_vars = parse_env()
updated = main(env_vars)

# Parse the command line and update the updated config accordingly
parse_args = ArgParser(epilog=EPILOG.format(updated))
actions, params = parse_args()
config = updated(params)
