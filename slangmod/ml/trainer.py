import torch.nn as ptn
import torch.optim as pto
import torch.optim.lr_scheduler as pts
from swak.funcflow import Curry
from swak.pt.train import Trainer, EpochPrinter, TrainPrinter, OnDisk
from swak.misc import StdOutLogger
from ..config import config

LOGGER = StdOutLogger(__name__, config.log_level)

checkpoint = OnDisk(config.checkpoint_file)
epoch_cb = EpochPrinter(LOGGER.debug)
train_cb = TrainPrinter(LOGGER.info)


loss = ptn.CrossEntropyLoss(
    ignore_index=0,
    label_smoothing=config.label_smoothing
)
#optimizer = Curry[pto.Adam](pto.Adam, config.learning_rate)
optimizer = Curry[pto.Adadelta](pto.Adadelta, 1.0)
scheduler = Curry[pts.ExponentialLR](pts.ExponentialLR, config.gamma)

trainer = Trainer(
    config.batch_size,
    config.max_epochs,
    loss,
    optimizer,
    scheduler,
    config.warmup,
    config.patience,
    config.max_n,
    checkpoint,
    epoch_cb,
    train_cb
)
