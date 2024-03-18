from ..base import ImageClassifyDataset, ClassifyDatasetConfig
import os 

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms 

class CIFAR10Config(ClassifyDatasetConfig):
    """
        Inherited member
        - batch_size 
        - usage 
    """
    def __init__(self, env_config):
        super().__init__(env_config)

    def dataset_root(self):
        subfolder = "cifar10"
        return os.path.join(
            self.env_config.dataset_root(), subfolder)

class CIFAR10(ImageClassifyDataset):
    def __init__(self, config: CIFAR10Config):
        super().__init__(config)
        self.name = "cifar10"
        
        will_download = not os.path.exists(config.dataset_root())
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            )
        ])
        self._dataset = datasets.CIFAR10(
            root=config.dataset_root(),
            train=config.usage == 'train',
            download=will_download,
            transform=transform
        )
        self._loader = DataLoader(self._dataset, batch_size=config.batch_size)  

    def classes(self):
        return [
            "plane", "car", "bird", "cat", "deer", 
            "dog", "frog", "horse", "ship", "truck"
        ]
    
    def __len__(self):
        return len(self._dataset)

    def __iter__(self):
        return iter(self._loader)

    def __getitem__(self, index: int):
        return self._dataset[index]

def create_cifar10(env_config, usage, batch_size=32):
    config = CIFAR10Config(env_config)
    config.usage = usage 
    config.batch_size = batch_size
    dataset = CIFAR10(config)
    return dataset