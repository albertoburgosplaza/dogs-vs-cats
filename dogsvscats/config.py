from pathlib import Path
import torch

DATA_PATH = Path("/workspaces/dogs-vs-cats/data")
TRAIN_PATH = DATA_PATH / "train"
TEST_PATH = DATA_PATH / "test1"
CLASSES = {0: "cat", 1: "dog"}
CLASSES_INV = {v: k for k, v in CLASSES.items()}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
