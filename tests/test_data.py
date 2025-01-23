import torch
from torch.utils.data import Dataset
import group83_mlops.data as dt


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = dt.MyDataset("data/raw")
    assert isinstance(dataset, Dataset)

def test_data():
    dt.preprocess_data("data/raw", "data/processed")
    main_dataset = dt.cifar100()
    main_dataloader = torch.utils.data.DataLoader(main_dataset, batch_size=64)

    assert type(main_dataset)== torch.Tensor , f"Expected main_dataset to be a torch.Tensor; however, it is of type {type(main_dataset)} "
    assert main_dataset.size(dim=1)  == 32, f'Expected main_dataset.size(dim=1)= 32; however, at dim = 1, has size {main_dataset.size(dim=1)}'
    assert main_dataset.size(dim=2)  == 32, f'Expected main_dataset.size(dim=2)= 32; however, at dim = 2, has size {main_dataset.size(dim=2)}'
    assert main_dataset.size(dim=3)  == 3, f'Expected main_dataset.size(dim=3)= 3; however, at dim = 3, has size {main_dataset.size(dim=2)}'

    print(main_dataset.size())
    print(main_dataset.mean(dim = (0,1,2)))
    print(main_dataset.std(dim = (0,1,2)))

    deviation = abs(torch.Tensor.mean(main_dataset))
    assert deviation <= 0.05 , f'Expected the data to be centered; however, the mean is {torch.Tensor.mean(main_dataset)}'
    deviation = abs(1 - torch.std(main_dataset))
    assert  deviation <= 0.05 , f'Expected the data to be normalised; however, the std is {torch.std(main_dataset)}'

    assert type(main_dataloader)== torch.utils.data.dataloader.DataLoader , f"Expected main_dataset to be a torch.Tensor; however, it is of type {type(main_dataloader)} "

    # assert True == False, 'flush print'
