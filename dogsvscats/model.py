import torch
from dogsvscats import config


def train_model(
    model, criterion, optimizer, scheduler, es, train_dl, valid_dl, num_epochs
):
    dataloaders = {"train": train_dl, "valid": valid_dl}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")

        for dl in dataloaders:
            model.train() if dl == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

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
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[dl].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[dl].dataset)

            print(f"{dl} -> Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if dl == "valid":
                scheduler.step(epoch_acc)
                es(epoch_acc, model)

        if es.early_stop:
            print("Early stopping")
            break

    return model
