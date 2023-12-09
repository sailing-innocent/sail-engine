from ..base import ImageClassifyDataset, ClassifyDatasetConfig
import os 

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class MNISTConfig(ClassifyDatasetConfig):
    """
        Inherited member
        - batch_size 
        - usage 
    """
    def __init__(self, env_config):
        super().__init__(env_config)

    def dataset_root(self):
        subfolder = "mnist"
        return os.path.join(
            self.env_config.dataset_root(), subfolder)

class MNIST(ImageClassifyDataset):
    def __init__(self, config: MNISTConfig):
        super().__init__(config)
        self.name = "mnist"
        will_download = not os.path.exists(config.dataset_root())
        self._dataset = datasets.MNIST(
            root=config.dataset_root(),
            train=config.usage == 'train',
            download=will_download,
            transform=ToTensor()
        )
        self._loader = DataLoader(self._dataset, batch_size=config.batch_size)

    def classes(self):
        return [
            "0", "1", "2", "3", "4", 
            "5", "6", "7", "8", "9"
        ]
    
    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        return iter(self._loader)

    def __getitem__(self, index: int):
        return self._dataset[index]

def create_mnist(env_config, usage, batch_size=32):
    mnist_config = MNISTConfig(env_config)
    mnist_config.usage = usage 
    mnist_config.batch_size = batch_size
    return MNIST(mnist_config)