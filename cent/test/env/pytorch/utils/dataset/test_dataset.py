import torch 
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __len__(self):
        return self.data_tensor.size(0)   
    
    def __getitem__(self, idx):
        return self.data_tensor[idx], self.target_tensor[idx]
    

def test_dataset():
    data_tensor = torch.randn(10, 3)
    target_tensor = torch.randint(2, (10, ))
    dataset = MyDataset(data_tensor, target_tensor)

    assert len(dataset) == 10
    for i in range(len(dataset)):
        data, target = dataset[i]
        print(data.shape)
        print(target.shape)
        break

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for data, target in dataloader:
        assert data.size(0) == 2
        assert target.size(0) == 2
        assert target[0].item() in [0, 1]
        break