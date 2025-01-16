from pathlib import Path
from tqdm import tqdm

import typer
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
import torchvision
import torchvision.transforms as transforms

from torch.profiler import profile, ProfilerActivity

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
        train_dataset = torchvision.datasets.CIFAR100(root=f'{self.data_path}', train=True, download=True, transform=transform)


        print('Prior to improvements')
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
            
            images = []
            
            for image, _ in tqdm(dataloader, desc="Processing CIFAR-100 train images"):
                images.append(image)

            images_tensor = torch.cat(images,dim=0)

            torch.save(images_tensor, f"{output_folder}/train_images.pt")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
        prof.export_chrome_trace("trace_prior.json")

        print('After improvements')
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            # dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
            dataloader = TensorDataset(torch.tensor(train_dataset.data).float() / 255.0 , torch.tensor(train_dataset.targets) )

            images = []  # Initialize an empty list to store images

            for image, _ in tqdm(dataloader, desc="Processing CIFAR-100 train images"):
                images.append(image)  # Append each image tensor to the list

            # Concatenate all image tensors along the 0th dimension (batch dimension)
            images_tensor = torch.cat(images, dim=0)

            # Save the resulting tensor to a file
            torch.save(images_tensor, f"{output_folder}/train_images.pt")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
        prof.export_chrome_trace("trace_posterior.json")

        # Do the same for the test data set
        test_dataset = torchvision.datasets.CIFAR100(root=f'{self.data_path}', train=False, download=True, transform=transform)
        # dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
        dataloader = TensorDataset(torch.tensor(test_dataset.data).float() / 255.0 , torch.tensor(test_dataset.targets) )

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
