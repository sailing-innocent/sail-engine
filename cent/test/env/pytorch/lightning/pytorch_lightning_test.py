import pytest

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from .hello import train, use

dataset = MNIST(
    "D:/data/datasets/mnist", 
    download=False, 
    transform=ToTensor())

@pytest.mark.func 
def test_train_lightning():
    train(dataset)    
    assert True

@pytest.mark.current
def test_train_lightning():
    use(dataset)
    assert True 