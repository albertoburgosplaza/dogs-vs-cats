from torch.utils.data import DataLoader
from dogsvscats.model import eval_model, load_model
from dogsvscats.dataset import get_datasets
from dogsvscats import config

train_ds, _, _ = get_datasets()
train_dl = DataLoader(train_ds, config.BS, shuffle=False, num_workers=config.NW)

model = load_model(config.MODEL_NAME, config.CHECKPOINT_PATH)
eval_model(model, train_dl)
