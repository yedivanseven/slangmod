from swak.funcflow import Fork
from swak.pt.types import Module
from swak.pt.io import ModelSaver, StateSaver, StateLoader
from ..config import config

__all__ = [
    'save_model',
    'load_model'
]

# Just so that we don't clutter the "steps" with definitions
save_model = Fork[[Module], tuple[()]](
    ModelSaver(config.model_file, create=True),
    StateSaver(config.weights_file, create=True)
)
load_model = StateLoader(config.weights_file)
