from module.dataset.base import BaseDataset, BaseDatasetConfig
from module.data.camera_info import CameraInfo 
from module.data.image_info import ImageInfo 

from typing import List 

from dataclasses import dataclass 
import numpy as np 

@dataclass 
class CameraImagePair:
    cam: CameraInfo
    img: ImageInfo

@dataclass 
class RayColorPair:
    orig: np.ndarray
    dir: np.ndarray
    color: np.ndarray

class NVSDatasetConfig(BaseDatasetConfig):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.scene_scale = 1.0
        self.scale = 1.0
        self.enable_rays = False 
        self.obj_name = "dummy"
        self.name = "nvs"
        self.obj_list = ["dummy"]

class NVSDataset(BaseDataset):
    def __init__(self, config: NVSDatasetConfig):
        super().__init__(config)
        self.N = 0
        self.name = config.name
        self.obj_name = config.obj_name
        # Camera-Image pairs
        self._cam_img_pairs = []
        # Ray - Color pairs
        self._ray_color_pairs = []
        self.use_rays = False

    def enable_rays(self):
        return self.config.enable_rays
    
    def __getitem__(self, idx):
        if (self.use_rays):
            return self._ray_color_pairs[idx]
        else:
            return self._cam_img_pairs[idx]
        
    def __iter__(self):
        if (self.use_rays):
            raise NotImplementedError("Not implemented")
        else:
            for _pair in self._cam_img_pairs:
                yield _pair.cam, _pair.img 

    def __len__(self):
        return self.N
    
    def pairs(self):
        return self._cam_img_pairs.copy()
    
    def _load_dataset(self):
        raise NotImplementedError("Not implemented")