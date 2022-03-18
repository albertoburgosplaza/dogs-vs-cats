import warnings
import argparse
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dogsvscats.data import get_datasets
from dogsvscats.model import train_model, load_model, MODELS
from dogsvscats.callbacks import EarlyStopping

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    choices=MODELS,
    default="mobilenet_v3_small",
    help="Model name",
    type=str,
)
parser.add_argument(
    "-cp",
    "--checkpoint-path",
    help="Checkpoint Path",
    type=str,
)
parser.add_argument("-w", "--workers", default=0, help="Workers", type=int)
parser.add_argument("-bs", "--batch-size", default=8, help="Batch size", type=int)
parser.add_argument(
    "-lr", "--learning-rate", default=0.01, help="Learning rate", type=float
)
parser.add_argument("-e", "--epochs", default=1, help="Epochs", type=int)

parser.add_argument(
    "-sp",
    "--scheduler-patience",
    default=3,
    help="Scheduler patience",
    type=int,
)
parser.add_argument(
    "-esp",
    "--early-stopping-patience",
    default=5,
    help="Early stopping patience",
    type=int,
)
parser.add_argument("-d", "--debug", default=False, help="Debug", action="store_true")
parser.add_argument(
    "-df", "--debug-fraction", default=0.05, help="Debug fraction", type=float
)
parser.add_argument(
    "-vf",
    "--validation-fraction",
    default=0.2,
    help="Validation fraction",
    type=float,
)
args = parser.parse_args()


train_ds, valid_ds, _ = get_datasets(
    valid_frac=args.validation_fraction,
    debug=args.debug,
    debug_frac=args.debug_fraction,
)

train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
valid_dl = DataLoader(
    valid_ds, args.batch_size, shuffle=False, num_workers=args.workers
)

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
