import torch.nn as ptn
import torch.optim as pto
import torch.optim.lr_scheduler as pts
from swak.funcflow import Curry
from swak.pt.train import Trainer, EpochPrinter, TrainPrinter, OnDisk
from swak.pt.train import LinearInverse, LinearCosine, LinearExponential
from swak.misc import StdOutLogger
from ..config import config, Scaling

LOGGER = StdOutLogger(__name__, config.log_level)

checkpoint = OnDisk(config.checkpoint_file, True)
epoch_cb = EpochPrinter(LOGGER.info)
train_cb = TrainPrinter(LOGGER.info)

loss = ptn.CrossEntropyLoss(
    ignore_index=0,
    label_smoothing=config.train.label_smoothing
)
optimizer = Curry[pto.Adam](pto.AdamW, config.train.learning_rate, fused=True)
inverse = LinearInverse(config.train.warmup, config.train.power)
exponential = LinearExponential(config.train.warmup, config.train.gamma)
cosine = LinearCosine(config.train.warmup, config.train.cooldown)
scaling = {
    Scaling.INVERSE: inverse,
    Scaling.EXPONENTIAL: exponential,
    Scaling.COSINE: cosine
}[config.train.scaling]
scheduler = Curry[pts.LambdaLR](pts.LambdaLR, scaling)

# ToDo: Log gradient norm for gradient clipping!
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
