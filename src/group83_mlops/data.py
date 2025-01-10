from pathlib import Path

import typer
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = torchvision.datasets.CIFAR100(root=f'{output_folder}', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root=f'{output_folder}', train=False, download=True, transform=transform)
        
        print(next(iter(train_dataset)))




def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
