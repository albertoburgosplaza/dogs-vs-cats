import argparse
import torch
import torch.onnx
import onnxruntime
import config
from dogsvscats.model import load_model, predict_model, MODELS
from dogsvscats.data import transform_ops
from dogsvscats.utils import pil_loader
from dogsvscats import config


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


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

torch.onnx.export(
    model,  # model being run
    image,  # model input (or a tuple for multiple inputs)
    f"{config.DATA_PATH / args.model}.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
        "input": {0: "batch_size"},  # variable length axes
        "output": {0: "batch_size"},
    },
)
