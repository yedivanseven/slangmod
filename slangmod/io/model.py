from swak.pt.io import ModelSaver
from ..config import config

__all__ = ['save_model']

save_model = ModelSaver(config.model_file, True)
