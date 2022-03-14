import warnings
import argparse
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dogsvscats.data import get_datasets
from dogsvscats.model import train_model, load_model, MODELS
from dogsvscats.callbacks import EarlyStopping
from dogsvscats import config

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    choices=MODELS,
    help="Model name",
    type=str,
)
parser.add_argument(
    "-cp",
    "--checkpoint-path",
    help="Checkpoint Path",
    type=str,
)
parser.add_argument("-w", "--workers", default=config.NW, help="Workers", type=int)
parser.add_argument(
    "-bs", "--batch-size", default=config.BS, help="Batch size", type=int
)
parser.add_argument(
    "-lr", "--learning-rate", default=config.LR, help="Learning rate", type=float
)
parser.add_argument("-e", "--epochs", default=config.EPOCHS, help="Epochs", type=int)

parser.add_argument(
    "-sp",
    "--scheduler-patience",
    default=config.SCHEDULER_PATIENCE,
    help="Scheduler patience",
    type=int,
)
parser.add_argument(
    "-esp",
    "--early-stopping-patience",
    default=config.EARLYSTOPPING_PATIENCE,
    help="Early stopping patience",
    type=int,
)
parser.add_argument("-d", "--debug", default=False, help="Debug", action="store_true")
parser.add_argument(
    "-df", "--debug-frac", default=0.05, help="Debug fraction", type=float
)
parser.add_argument(
    "-vf",
    "--valid-frac",
    default=config.VALID_FRAC,
    help="Validation fraction",
    type=float,
)
args = parser.parse_args()


train_ds, valid_ds, _ = get_datasets(
    valid_frac=args.valid_frac, debug=args.debug, debug_frac=args.debug_frac
)

train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
valid_dl = DataLoader(valid_ds, args.batch_size, shuffle=True, num_workers=args.workers)

model = load_model(args.model)

optimizer = optim.SGD(model.parameters(), args.learning_rate, momentum=0.9)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=args.scheduler_patience, verbose=True
)
es = EarlyStopping(
    patience=args.early_stopping_patience,
    mode="max",
    verbose=True,
    path=args.checkpoint_path,
)

model = train_model(model, optimizer, scheduler, es, train_dl, valid_dl, args.epochs)
