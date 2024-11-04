import torch.nn as ptn
import torch.optim as pto
import torch.optim.lr_scheduler as pts
from swak.funcflow import Curry
from swak.pt.train import Trainer, EpochPrinter, TrainPrinter, OnDisk
from swak.pt.train import LinearInverse
from swak.misc import StdOutLogger
from ..config import config

LOGGER = StdOutLogger(__name__, config.log_level)

checkpoint = OnDisk(config.files.checkpoint)
epoch_cb = EpochPrinter(LOGGER.info)
train_cb = TrainPrinter(LOGGER.info)

loss = ptn.CrossEntropyLoss(
    ignore_index=0,
    label_smoothing=config.train.label_smoothing
)
optimizer = Curry[pto.Adam](pto.Adam, config.lr, fused=True)
linear_inverse = LinearInverse(config.train.warmup, config.train.power)
scheduler = Curry[pts.LambdaLR](pts.LambdaLR, linear_inverse)

trainer = Trainer(
    loss=loss,
    optimizer=optimizer,
    batch_size=config.train.batch_size,
    max_epochs=config.train.max_epochs,
    scheduler=scheduler,
    warmup=config.train.warmup,
    patience=config.train.patience,
    checkpoint=checkpoint,
    step_freq=config.train.step_freq,
    epoch_cb=epoch_cb,
    train_cb=train_cb
)
