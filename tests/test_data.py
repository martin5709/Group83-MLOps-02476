import torch
from torch.utils.data import Dataset

from src.group83_mlops.data import MyDataset,cifar100

here ='here'

def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)

def test_data():
    
    main_dataset = cifar100()
    main_dataloader = torch.utils.data.DataLoader(main_dataset, batch_size=64)
    
    # assert len(dataset) == N_train for training and N_test for test
    assert type(main_dataset)== torch.Tensor , f"Expected main_dataset to be a torch.Tensor; however, it is of type {type(main_dataset)} "
    assert main_dataset.size(dim=1)  == 32, f'Expected main_dataset.size(dim=1)= 32; however, at dim = 1, has size {main_dataset.size(dim=1)}'
    assert main_dataset.size(dim=2)  == 3, f'Expected main_dataset.size(dim=1)= 32; however, at dim = 2, has size {main_dataset.size(dim=2)}'

    deviation = abs(0.5 - torch.Tensor.mean(main_dataset))/0.5
    assert deviation <= 0.05 , f'Expected the data to be centered; however, the mean is {torch.Tensor.mean(main_dataset)}'
    assert torch.std(main_dataset) <= 0.5 , f'Expected the data to be normalised; however, the std is {torch.std(main_dataset)}'

    assert type(main_dataloader)== torch.utils.data.dataloader.DataLoader , f"Expected main_dataset to be a torch.Tensor; however, it is of type {type(main_dataloader)} "







