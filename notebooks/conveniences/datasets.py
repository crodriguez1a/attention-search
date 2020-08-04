import os
from dataclasses import dataclass

import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

TV_TRANSFORM: transforms.Compose = transforms.Compose(
    [transforms.CenterCrop(32), transforms.ToTensor()]
    # [transforms.ToTensor()]
)

DATA_PATH: str = os.getenv("DATA_PATH", os.getcwd() + "/notebooks/data/")

TV_DATASETS: dict = {
    "flickr_8k": {
        "name": "flickr_8k",
        "description": "",
        "module": torchvision.datasets.Flickr8k,
        "args": [
            f"{DATA_PATH}/Flicker8k/Flicker8k_Dataset/",
            f"{DATA_PATH}/Flickr8k/Flickr8k_text/Flickr8k.token.txt",
        ],
        "kwargs": {"transform": TV_TRANSFORM},
    },
    "svhn": {
        "name": "svhn",
        "description": "",
        "module": torchvision.datasets.SVHN,
        "args": [f"{DATA_PATH}/SVHN/"],
        "kwargs": {"split": "train", "transform": TV_TRANSFORM, "download": False},
    },
    "sbu": {
        "name": "sbu",
        "description": "",
        "module": torchvision.datasets.SBU,
        "args": [f"{DATA_PATH}/SBU/"],
        "kwargs": {"split": "train", "transform": TV_TRANSFORM, "download": False},
    },
    "voc": {
        "name": "voc",
        "description": "",
        "module": torchvision.datasets.VOCSegmentation,
        "args": [f"{DATA_PATH}/VOC/"],
        "kwargs": {"split": "train", "transform": TV_TRANSFORM, "download": False},
    },
    "usps": {
        "name": "usps",
        "description": "",
        "module": torchvision.datasets.USPS,
        "args": [f"{DATA_PATH}/USPS/"],
        "kwargs": {
            "year": "2007",
            "image_set": "train",
            "download": True,
            "transform": TV_TRANSFORM,
        },
    },
    "celeb_a": {
        "name": "celeb_a",
        "description": "",
        "module": torchvision.datasets.CelebA,
        "args": [f"{DATA_PATH}/CelebA/"],
        "kwargs": {"download": True, "transform": TV_TRANSFORM,},
    },
}


@dataclass
class DatasetMeta:
    name: str
    description: str
    module: object
    args: list
    kwargs: dict


def load_dataset(name: str) -> torch.utils.data.DataLoader:
    ds: DatasetMeta = DatasetMeta(**TV_DATASETS.get(name))
    dataset = ds.module(*ds.args, **ds.kwargs)
    return torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)


def get_vectors(
    dataset_size: int, data_loader: torch.utils.data.DataLoader
) -> np.ndarray:
    labels: list = []
    vectors: list = []
    for i in tqdm(range(dataset_size)):
        arr, label = data_loader.dataset.__getitem__(i)
        labels.append(label)
        vectors.append(arr.numpy())

    # TODO: fix this
    vecs = np.array([i[:, ...] for i in vectors])
    breakpoint()
    return vecs, labels


if __name__ == "__main__":
    # ds = "celeb_a"
    ds = "svhn"
    data_loader: torch.utils.data.DataLoader = load_dataset(ds)
    # TODO: Dynamic size
    dataset_size: int = data_loader.dataset.data.shape[0]
    # dataset_size = data_loader.dataset.bbox.shape[0]
    vectors, labels = get_vectors(dataset_size, data_loader)

    print("Saving vectors and labels...")
    np.save(f"{DATA_PATH}/{ds}", vectors)

    try:
        np.save(f"{DATA_PATH}/{ds}_labels", labels)
    except:
        np.save(f"{DATA_PATH}/{ds}_labels", range(dataset_size))

    print("Saved.")
