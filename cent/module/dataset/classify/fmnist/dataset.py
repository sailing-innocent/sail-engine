from ..base import ImageClassifyDataset, ClassifyDatasetConfig
import os 

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class FMNISTConfig(ClassifyDatasetConfig):
    """
        Inherited member
        - batch_size 
        - usage 
    """
    def __init__(self, env_config):
        super().__init__(env_config)

    def dataset_root(self):
        subfolder = "fmnist"
        return os.path.join(
            self.env_config.dataset_root, subfolder)

class FMNIST(ImageClassifyDataset):
    def __init__(self, config: FMNISTConfig):
        super().__init__(config)
        self.name = "fmnist"
        
        will_download = not os.path.exists(config.dataset_root())
        self._dataset = datasets.FashionMNIST(
            root=config.dataset_root(),
            train=config.usage == 'train',
            download=will_download,
            transform=ToTensor()
        )
        self._loader = DataLoader(self._dataset, batch_size=config.batch_size)

    def classes(self):
        return [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot"
        ]

    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        return iter(self._loader)

    def __getitem__(self, index: int):
        return self._dataset[index]

def create_dataset(env_config, usage, batch_size=32):
    fmnist_config = FMNISTConfig(env_config)
    fmnist_config.usage = usage 
    fmnist_config.batch_size = batch_size
    return FMNIST(fmnist_config)