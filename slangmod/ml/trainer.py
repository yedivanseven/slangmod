import torch.optim as pto
import torch.optim.lr_scheduler as pts
from swak.funcflow import Curry
from swak.pt.train import (
    Trainer,
    StepPrinter,
    EpochPrinter,
    TrainPrinter,
    OnDisk
)
from swak.pt.train import LinearInverse, LinearCosine, LinearExponential
from swak.pt.losses import XEntropyLoss
from swak.misc import StdLogger, FileLogger, RAW_FMT, SHORT_FMT
from ..config import config, Optimizers, Scaling
from ..io import save_train_toml
from .tokenizers import special

__all__ = [
    'trainer',
    'criterion'
]

# Create various Loggers
LOG_TERM = StdLogger(__name__, config.log_level)
LOG_FILE = FileLogger(config.log_file, fmt=SHORT_FMT, mode=config.mode)
MONITOR = FileLogger(config.monitor_file, fmt=RAW_FMT, mode=config.mode)

# Initialize a model checkpoint
checkpoint = OnDisk(config.checkpoint_file, create=True)

# Configure callbacks every step, every epoch, and at the end of training.
step_cbs = StepPrinter(MONITOR.debug),
epoch_cbs = EpochPrinter(LOG_TERM.info), EpochPrinter(LOG_FILE.info)
train_cbs = (
    TrainPrinter(LOG_TERM.info),
    TrainPrinter(LOG_FILE.info),
    save_train_toml
)

# Initialize the loss function
criterion = XEntropyLoss(
    ignore_index=special.pad_id,
    label_smoothing=config.train.label_smoothing
)

# Initialize two options for the optimizer ...
adamw = Curry[pto.AdamW](pto.AdamW, config.train.learning_rate, fused=True)
adafactor = Curry[pto.Adafactor](pto.Adafactor, config.train.learning_rate)
# ... and pick according to configuration.
optimizer = {
    Optimizers.ADMAW: adamw,
    Optimizers.ADAFACTOR: adafactor
}[config.train.optimizer]

# Initialize various learning-rate schedulers ...
inverse = LinearInverse(config.train.warmup, config.train.power)
exponential = LinearExponential(config.train.warmup, config.train.gamma)
cosine = LinearCosine(config.train.warmup, config.train.cooldown)
# ... and select according to config.
scaling = {
    Scaling.INVERSE: inverse,
    Scaling.EXPONENTIAL: exponential,
    Scaling.COSINE: cosine
}[config.train.scaling]
scheduler = Curry[pts.LambdaLR](pts.LambdaLR, scaling)

# Create the training loop
trainer = Trainer(
    loss=criterion,
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
    show_progress=config.progress,
    step_cbs=step_cbs,
    cb_freq=config.train.cb_freq,
    epoch_cbs=epoch_cbs,
    train_cbs=train_cbs
)
