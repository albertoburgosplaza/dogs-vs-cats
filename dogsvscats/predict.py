import argparse
import warnings
import torch
from dogsvscats.model import load_model, predict_model, MODELS
from dogsvscats.data import transform_ops
from dogsvscats.utils import pil_loader
from dogsvscats import config

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="Input image")
parser.add_argument(
    "-m", "--model", default=config.MODEL_NAME, choices=MODELS, help="Model name"
)
parser.add_argument("-cp", "--checkpoint-path", default=None, help="Checkpoint Path")
parser.add_argument(
    "-d",
    "--download",
    default=False,
    help="Checkpoint Path",
    action="store_true",
)
args = parser.parse_args()

image = torch.unsqueeze(transform_ops(pil_loader(args.image)), 0).to(config.DEVICE)

model = load_model(args.model, args.checkpoint_path, args.download)

pred_class = predict_model(model, image)

print(config.CLASSES[pred_class])
