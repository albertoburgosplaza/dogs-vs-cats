import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision import models
from dogsvscats.metrics import Metrics
from dogsvscats import config

MODELS = {
    "resnet18": "https://github.com/albertoburgosplaza/dogs-vs-cats/releases/download/pytorch-v0.5.1/resnet18.pt",
    "mobilenet_v3_small": "https://github.com/albertoburgosplaza/dogs-vs-cats/releases/download/pytorch-v0.5.1/mobilenet_v3_small.pt",  # noqa
}


def load_model(model_name: str, checkpoint_path=None, download=False):
    if model_name not in MODELS:
        raise ValueError(f"{model_name} not available.")

    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(config.CLASSES))
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=True)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, len(config.CLASSES))

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
    elif download:
        model.load_state_dict(
            model_zoo.load_url(MODELS[model_name], progress=True, map_location="cpu")
        )

    model = model.to(config.DEVICE)
    return model


def train_one_epoch(model, optimizer, scheduler, es, dataloaders):
    for dl, value in dataloaders.items():
        print(f"Processing {dl} data...")
        model.train() if dl == "train" else model.eval()
        metrics = Metrics()

        for inputs, labels in value:
            inputs = inputs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            metrics(outputs, labels, inputs.size(0))

            if dl == "train":
                metrics.loss.batch_loss.backward()
                optimizer.step()

        metrics.show()

        if dl == "valid":
            scheduler.step(metrics.accuracy.get())
            es(metrics.accuracy.get(), model)


def train_model(model, optimizer, scheduler, es, train_dl, valid_dl, num_epochs):
    dataloaders = {"train": train_dl, "valid": valid_dl}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")

        train_one_epoch(model, optimizer, scheduler, es, dataloaders)

        if es.early_stop:
            print("Early stopping")
            break

    return model


def eval_model(model, dl):
    model.eval()
    metrics = Metrics()

    for inputs, labels in dl:
        inputs = inputs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        outputs = model(inputs)
        metrics(outputs, labels, inputs.size(0))

    metrics.show()


def predict_model(model, image):
    model.eval()
    output = model(image)
    _, pred = torch.max(output, 1)
    return int(pred.cpu().item())
