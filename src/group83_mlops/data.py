from pathlib import Path

import typer
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


RAW_DATA_PATH="./data/raw"
OUT_DATA ="./data/processed"

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path
        # self.dataset = None

    def __len__(self) -> int:
        """Return the length of the dataset."""
        # return len(self.dataset)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        # image, label = self.train_dataset[index]
        # return image, label
        

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        #Because cifar-100 provides PIL-images we need to do .Totensor() instead of tensordataset(..)
        transform = transforms.Compose([      
            transforms.ToTensor(),   # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5),  # Normalize to [-1, 1] for GANs
                                 (0.5, 0.5, 0.5))
        ])  
        print("<<preprocessing>>")
        #Normally these two lines would be enough, but we are trying "good practices" so I will store them unnecessarily.
        train_dataset = torchvision.datasets.CIFAR100(root=f'{self.data_path}', train=True, download=True, transform=transform)
        
        dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
        # print(next(iter(dataloader)))
        images = []
        for image, _ in tqdm(dataloader, desc="Processing CIFAR-100 train images"):
            images.append(image)

        images_tensor = torch.cat(images,dim=0)

        torch.save(images_tensor, f"{output_folder}/train_images.pt")

        # Do the same for the test data set
        test_dataset = torchvision.datasets.CIFAR100(root=f'{self.data_path}', train=False, download=True, transform=transform)
        dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
        images = []
        for image, _ in tqdm(dataloader, desc="Processing CIFAR-100 test images"):
            images.append(image)

        images_tensor = torch.cat(images,dim=0)

        torch.save(images_tensor, f"{output_folder}/test_images.pt")


def preprocess_data(raw_data_path: Path, output_data_path: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_data_path)


def cifar100() -> tuple[torch.utils.data.Dataset]:
    """Return train and test datasets for cifar-100."""
    #assumes self.data_path = data/raw/cifar-100-python
    dataset = torch.load(f"{OUT_DATA}/train_images.pt")
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    # dataset = TensorDataset(dataset)
    return dataset

if __name__ == "__main__":
    typer.run(preprocess_data)
    # typer.run(cifar100())
    dataloader = DataLoader(typer.run(cifar100()), batch_size=64, shuffle=True, num_workers=2)
    print(next(iter(dataloader)))
