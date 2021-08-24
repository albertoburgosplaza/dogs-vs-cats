from typing import List
import numpy as np
import torch
from torch import nn
import torchmetrics as tm
from dogsvscats import config


class Accuracy:
    def __init__(self) -> None:
        self.metric = tm.Accuracy()

    def __call__(self, predictions, labels):
        self.metric(predictions, labels)

    def get(self):
        return self.metric.compute()

    def show(self):
        print(f"Accuracy: {self.get():.4f}")


class Precision:
    def __init__(self) -> None:
        self.metric = tm.Precision(num_classes=len(config.CLASSES), average="none")

    def __call__(self, predictions, labels):
        self.metric(predictions, labels)

    def get(self):
        return self.metric.compute()

    def show(self):
        print(f"Precision: {[round(v, 4) for v in self.get().tolist()]}")


class Recall:
    def __init__(self) -> None:
        self.metric = tm.Recall(num_classes=len(config.CLASSES), average="none")

    def __call__(self, predictions, labels):
        self.metric(predictions, labels)

    def get(self):
        return self.metric.compute()

    def show(self):
        print(f"Recall: {[round(v, 4) for v in self.get().tolist()]}")


class Loss:
    def __init__(self) -> None:
        self.metric = nn.CrossEntropyLoss()
        self.losses: List[float] = []
        self.batch_loss = None

    def __call__(self, outputs, labels, inputs_size):
        self.batch_loss = self.metric(outputs, labels)
        self.losses.append(self.batch_loss.item() * inputs_size)

    def get(self):
        return np.mean(self.losses)

    def show(self):
        print(f"Loss: {self.get():.4f}")


class Metrics:
    def __init__(self) -> None:
        self.loss = Loss()
        self.accuracy = Accuracy()
        self.precision = Precision()
        self.recall = Recall()

    def __call__(self, outputs, labels, inputs_size):
        outputs = outputs.cpu()
        labels = labels.cpu()
        _, preds = torch.max(outputs, 1)
        self.loss(outputs, labels, inputs_size)
        self.accuracy(preds, labels.data)
        self.precision(preds, labels.data)
        self.recall(preds, labels.data)

    def show(self):
        self.loss.show()
        self.accuracy.show()
        self.precision.show()
        self.recall.show()
