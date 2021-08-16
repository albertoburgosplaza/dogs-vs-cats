import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dogsvscats.dataset import get_datasets
from dogsvscats.model import train_model, load_model
from dogsvscats.callbacks import EarlyStopping
from dogsvscats import config


def train(batch_size=config.BS, lr=config.LR, num_epochs=config.EPOCHS):
    train_ds, valid_ds, _ = get_datasets()
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=config.NW)
    valid_dl = DataLoader(valid_ds, batch_size, shuffle=True, num_workers=config.NW)

    model = load_model(config.MODEL_NAME)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=config.SCHEDULER_PATIENCE, verbose=True
    )
    es = EarlyStopping(
        patience=config.EARLYSTOPPING_PATIENCE,
        mode="max",
        verbose=True,
        path=config.DATA_PATH / "checkpoint.pt",
    )

    model = train_model(
        model, criterion, optimizer, scheduler, es, train_dl, valid_dl, num_epochs
    )


train()
