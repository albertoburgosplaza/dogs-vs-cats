from pathlib import Path

DATA_PATH = Path("/home/alberto/workspace/dogs-vs-cats/data")
TRAIN_PATH = DATA_PATH / "train"
TEST_PATH = DATA_PATH / "test1"
CLASSES = {0: "cat", 1: "dog"}
CLASSES_INV = {v: k for k, v in CLASSES.items()}
DEVICE = "cuda"
# DEVICE = "cpu"
VALID_FRAC = 0
EPOCHS = 20
LR = 0.001
BS = 128
NW = 6
SCHEDULER_PATIENCE = 2
EARLYSTOPPING_PATIENCE = 4
MODEL_NAME = "mobilenet_v3_small"
CHECKPOINT_PATH = DATA_PATH / "mobilenet_v3_small.pt"