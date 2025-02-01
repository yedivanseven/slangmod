from swak.pt.io import ModelSaver, ModelLoader
from ..config import config

__all__ = [
    'save_model',
    'load_model'
]

save_model = ModelSaver(config.model_file, create=True)
load_model = ModelLoader(config.model_file)
