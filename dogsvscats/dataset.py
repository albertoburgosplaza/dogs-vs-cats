from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from dogsvscats import config
from dogsvscats.utils import pil_loader

transform_ops = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_datasets():
    train_data = {"path": [], "name": [], "id": [], "label": []}

    for f in config.TRAIN_PATH.glob(pattern="*"):
        train_data["path"].append(str(f))
        name = str(f).split("/")[-1]
        train_data["name"].append(name)
        train_data["label"].append(config.CLASSES_INV[name.split(".")[0]])
        train_data["id"].append(name.split(".")[1])

    train_data = pd.DataFrame(train_data)

    if config.DEBUG:
        train_data = train_data.sample(frac=config.DEBUG_FRAC).reset_index(drop=True)

    valid_idx = train_data.sample(frac=config.VALID_SAMPLE).index
    valid_data = train_data.loc[valid_idx].reset_index(drop=True)
    train_data = train_data.loc[train_data.index.difference(valid_idx)].reset_index(
        drop=True
    )

    train_ds = DogsVSCats(train_data, False, transform_ops)
    valid_ds = DogsVSCats(valid_data, False, transform_ops)

    test_data = {"path": [], "name": [], "id": [], "label": []}

    for f in config.TEST_PATH.glob(pattern="*"):
        test_data["path"].append(str(f))
        name = str(f).split("/")[-1]
        test_data["name"].append(name)
        test_data["label"].append(-1)  # Non defined class
        test_data["id"].append(name.split(".")[1])

    test_data = pd.DataFrame(test_data)
    test_ds = DogsVSCats(test_data, True, transform_ops)

    return train_ds, valid_ds, test_ds


class DogsVSCats(Dataset):
    def __init__(self, df, test=False, transform_ops=False) -> None:
        super().__init__()
        self.df = df
        self.test = test
        self.transform_ops = transform_ops

    def __getitem__(self, index):
        image = self.transform_ops(pil_loader(self.df.loc[index].path))
        label = self.df.loc[index].label

        if self.test:
            return image
        return image, label

    def __len__(self) -> int:
        return len(self.df)
