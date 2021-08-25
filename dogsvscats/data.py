from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split
from dogsvscats import config
from dogsvscats.utils import pil_loader

transform_ops = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_datasets(valid_frac, debug=False, debug_frac=None):
    train_data = {"path": [], "name": [], "id": []}
    train_labels = []

    for f in config.TRAIN_PATH.glob(pattern="*"):
        train_data["path"].append(str(f))
        name = str(f).split("/")[-1]
        train_data["name"].append(name)
        train_data["id"].append(name.split(".")[1])
        train_labels.append(config.CLASSES_INV[name.split(".")[0]])

    train_data = pd.DataFrame(train_data)
    train_labels = pd.Series(train_labels)

    if debug:
        print("Debug mode...")
        train_data, _, train_labels, _ = train_test_split(
            train_data,
            train_labels,
            train_size=debug_frac,
            stratify=train_labels,
        )

    valid_ds = None

    if valid_frac > 0:
        train_data, valid_data, train_labels, valid_labels = train_test_split(
            train_data,
            train_labels,
            test_size=valid_frac,
            stratify=train_labels,
        )

        valid_ds = DogsVSCats(valid_data, valid_labels, transform_ops)

    train_ds = DogsVSCats(train_data, train_labels, transform_ops)

    test_data = {"path": [], "name": [], "id": []}

    for f in config.TEST_PATH.glob(pattern="*"):
        test_data["path"].append(str(f))
        name = str(f).split("/")[-1]
        test_data["name"].append(name)
        test_data["id"].append(name.split(".")[1])

    test_data = pd.DataFrame(test_data)
    test_ds = DogsVSCats(test_data, transform_ops=transform_ops)

    return train_ds, valid_ds, test_ds


class DogsVSCats(Dataset):
    def __init__(
        self, data: pd.DataFrame, labels: pd.Series = None, transform_ops=False
    ) -> None:
        super().__init__()
        self.data = data.reset_index()
        if labels is not None:
            self.labels = labels.reset_index(drop=True)
        self.transform_ops = transform_ops

    def __getitem__(self, index):
        image = self.transform_ops(pil_loader(self.data.loc[index].path))

        if self.labels is not None:
            label = self.labels[index]
            return image, label
        return image

    def __len__(self) -> int:
        return len(self.data)
