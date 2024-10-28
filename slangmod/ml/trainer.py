import torch.nn as ptn
import torch.optim as pto
import torch.optim.lr_scheduler as pts
from swak.funcflow import Curry
from swak.pt.train import Trainer, EpochPrinter, TrainPrinter, OnDisk
from swak.misc import StdOutLogger
from ..config import config

LOGGER = StdOutLogger(__name__, config.log_level)

checkpoint = OnDisk(config.files.checkpoint)
epoch_cb = EpochPrinter(LOGGER.debug)
train_cb = TrainPrinter(LOGGER.info)

loss = ptn.CrossEntropyLoss(
    ignore_index=0,
    label_smoothing=config.train.label_smoothing
)
optimizer = Curry[pto.Adam](pto.Adam, config.train.learning_rate, fused=True)
scheduler = Curry[pts.LambdaLR](pts.LambdaLR, lambda epoch: (1 + epoch)**-0.5)

trainer = Trainer(
    batch_size=config.train.batch_size,
    max_epochs=config.train.max_epochs,
    loss=loss,
    optimizer=optimizer,
    scheduler=scheduler,
    warmup=config.train.warmup,
    patience=config.train.patience,
    max_n=config.train.max_n,
    checkpoint=checkpoint,
    epoch_cb=epoch_cb,
    train_cb=train_cb
)
