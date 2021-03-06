import warnings
import argparse
from torch.utils.data import DataLoader
from dogsvscats.model import eval_model, load_model, MODELS
from dogsvscats.data import get_datasets
from dogsvscats import config

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--model", choices=MODELS, help="Model name"
)
parser.add_argument(
    "-cp", "--checkpoint-path", help="Checkpoint Path"
)
parser.add_argument("-w", "--workers", default=config.NW, help="Workers", type=int)
parser.add_argument(
    "-bs", "--batch-size", default=config.BS, help="Batch size", type=int
)
args = parser.parse_args()

train_ds, _, _ = get_datasets(valid_frac=0)
train_dl = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
)

model = load_model(args.model, args.checkpoint_path)
eval_model(model, train_dl)
