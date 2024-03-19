from module.dataset.base import BaseDataset, BaseDatasetConfig
from abc import ABC, abstractmethod

class ClassifyDatasetConfig(BaseDatasetConfig):
    def __init__(self, env_config):
        super().__init__(env_config)

    def dataset_root(self):
        return ""

class ClassifyDataset(BaseDataset, ABC):
    def __init__(self, config: ClassifyDatasetConfig):
        super().__init__(config)

    @abstractmethod
    def classes(self):
        return []

    def __len__(self):
        return 0

    def __getitem__(self, index):
        pass

class ImageClassifyDataset(ClassifyDataset):
    def __init__(self, config: ClassifyDatasetConfig):
        super().__init__(config)
