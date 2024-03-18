import os  
from module.dataset.nvs.nsvf.dataset import NSVFDatasetConfig, NSVFDataset

class TankTempleDatasetConfig(NSVFDatasetConfig):
    """
    Inherited
        - env_config
        - batch_size = 64
        - usage = "train"
    """
    def __init__(self, env_config):
        super().__init__(env_config)
        self.name = "tank_temple"
        self.obj_name = "Train"
        self.obj_list = ["M60", "Playground", "Train", "Truck"]
        self.images = "images_4"
        self.white_bkgd = True
        self.llffhold = 8

    def dataset_root(self):
        return os.path.join(self.env_config.dataset_root, "TanksAndTempleBG", self.obj_name)

class TankTempleDataset(NSVFDataset):
    def __init__(self, config: TankTempleDatasetConfig):
        super().__init__(config)
        self._load_dataset()

def create_dataset(env_config, obj_name, usage):
    config = TankTempleDatasetConfig(env_config)
    config.usage = usage 
    config.obj_name = obj_name 
    return TankTempleDataset(config)