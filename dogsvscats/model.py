import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import Accuracy, Precision, Recall
from dogsvscats import config

MODELS = ["resnet18", "mobilenet_v3_small"]


def load_model(model_name: str, checkpoint_path=None):
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

    model = model.to(config.DEVICE)
    return model


def train_model(
    model, criterion, optimizer, scheduler, es, train_dl, valid_dl, num_epochs
):
    dataloaders = {"train": train_dl, "valid": valid_dl}

    accuracy = {dl: Accuracy().to(config.DEVICE) for dl in dataloaders}

    precision = {
        dl: Precision(num_classes=len(config.CLASSES), average=None).to(config.DEVICE)
        for dl in dataloaders
    }

    recall = {
        dl: Recall(num_classes=len(config.CLASSES), average=None).to(config.DEVICE)
        for dl in dataloaders
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")

        for dl in dataloaders:
            model.train() if dl == "train" else model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[dl]:
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if dl == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                accuracy[dl](preds, labels.data)
                precision[dl](preds, labels.data)
                recall[dl](preds, labels.data)

            epoch_loss = running_loss / len(dataloaders[dl].dataset)

            print(
                f"{dl} -> Loss: {epoch_loss:.4f} "
                f"Accuracy: {accuracy[dl].compute():.4f} "
                f"Precision: {[round(v, 4) for v in precision[dl].compute().tolist()]} "
                f"Recall: {[round(v, 4) for v in recall[dl].compute().tolist()]}"
            )

            if dl == "valid":
                scheduler.step(accuracy[dl].compute())
                es(accuracy[dl].compute(), model)

        if es.early_stop:
            print("Early stopping")
            break

    return model


def eval_model(model, dl):
    model.eval()

    accuracy = Accuracy().to(config.DEVICE)
    precision = Precision(num_classes=len(config.CLASSES), average=None).to(
        config.DEVICE
    )
    recall = Recall(num_classes=len(config.CLASSES), average=None).to(config.DEVICE)

    for inputs, labels in dl:
        inputs = inputs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        accuracy(preds, labels.data)
        precision(preds, labels.data)
        recall(preds, labels.data)

    print(
        f"Accuracy: {accuracy.compute():.4f} "
        f"Precision: {[round(v, 4) for v in precision.compute().tolist()]} "
        f"Recall: {[round(v, 4) for v in recall.compute().tolist()]}"
    )
