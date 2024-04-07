from module.dataset.nvs.colmap.dataset import ColmapDatasetConfig, ColmapDataset
import os 
class Mip360DatasetConfig(ColmapDatasetConfig):
    """
    Inherited
        - env_config
        - batch_size = 64
        - usage = "train"
    """
    def __init__(self, env_config):
        super().__init__(env_config)
        self.name = "mip360"
        self.obj_name = "bicycle"
        self.obj_list = ['bicycle', 'bonsai', 'counter','garden', 'kitchen', 'room', 'stump']
        self.images = "images_4"
        self.white_bkgd = True
        self.type = "RGBD"
        self.llffhold = 8

    def dataset_root(self):
        return os.path.join(self.env_config.dataset_root, "mip360d", self.obj_name)

class Mip360Dataset(ColmapDataset):
    def __init__(self, config: Mip360DatasetConfig):
        super().__init__(config)
        self._load_dataset()

def create_dataset(env_config, obj_name, usage):
    config = Mip360DatasetConfig(env_config)
    config.usage = usage 
    config.obj_name = obj_name 
    return Mip360Dataset(config)