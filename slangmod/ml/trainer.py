import torch.optim as pto
import torch.optim.lr_scheduler as pts
from swak.funcflow import Curry
from swak.pt.train import Trainer, EpochPrinter, TrainPrinter, OnDisk
from swak.pt.train import LinearInverse, LinearCosine, LinearExponential
from swak.pt.losses import XEntropyLoss
from swak.misc import StdOutLogger
from ..config import config, Optimizers, Scaling
from .tokenizers import special

LOGGER = StdOutLogger(__name__, config.log_level)

checkpoint = OnDisk(config.checkpoint_file, create=True)
epoch_cb = EpochPrinter(LOGGER.info)
train_cb = TrainPrinter(LOGGER.info)

loss = XEntropyLoss(
    ignore_index=special.pad_id,
    label_smoothing=config.train.label_smoothing
)

adamw = Curry[pto.AdamW](pto.AdamW, config.train.learning_rate, fused=True)
adafactor = Curry[pto.Adafactor](pto.Adafactor, config.train.learning_rate)
optimizer = {
    Optimizers.ADMAW: adamw,
    Optimizers.ADAFACTOR: adafactor
}[config.train.optimizer]

inverse = LinearInverse(config.train.warmup, config.train.power)
exponential = LinearExponential(config.train.warmup, config.train.gamma)
cosine = LinearCosine(config.train.warmup, config.train.cooldown)
scaling = {
    Scaling.INVERSE: inverse,
    Scaling.EXPONENTIAL: exponential,
    Scaling.COSINE: cosine
}[config.train.scaling]
scheduler = Curry[pts.LambdaLR](pts.LambdaLR, scaling)

trainer = Trainer(
    loss=loss,
    optimizer=optimizer,
    batch_size=config.train.batch_size,
    max_epochs=config.train.max_epochs,
    scheduler=scheduler,
    warmup=config.train.warmup,
    batch_step=True,
    patience=config.train.patience,
    step_freq=config.train.step_freq,
    clip_grad=config.train.clip_grad,
    checkpoint=checkpoint,
    epoch_cb=epoch_cb,
    train_cb=train_cb
)
