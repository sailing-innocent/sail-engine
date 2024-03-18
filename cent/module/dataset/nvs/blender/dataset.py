from ..base import NVSDatasetConfig, NVSDataset
from ..base import CameraImagePair, RayColorPair

from module.utils.image.basic import Image
from module.data.image_info import ImageInfo
from module.data.camera_info import CameraInfo

import os 
import json 
import numpy as np 
from tqdm import tqdm 

from module.utils.pointcloud.io import fetchPly

class NeRFBlenderDatasetConfig(NVSDatasetConfig):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.white_bkgd = True 
        self.use_point_list = False 
        self.obj_name = "lego"
        self.obj_list = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
        self.half_res = False 
        self.scene_scale = 1
        self.resw = 800
        self.resh = 800

    def dataset_root(self):
        return os.path.join(
            self.env_config.dataset_root,"nerf_synthetic", 
            self.obj_name)
    
class NeRFBlenderDataset(NVSDataset):
    """
    Inherited
        - config
    """
    def __init__(self, config: NeRFBlenderDatasetConfig):
        super().__init__(config)
        self.name = "nerf_blender"
        self._load_dataset()

    def _load_dataset(self):
        # load dataset 
        meta_file_path = os.path.join(
            self.config.dataset_root(), "transforms_{}.json".format(self.config.usage))
        
        with open(meta_file_path, 'r') as f:
            meta = json.load(f)
    
        fov_x = meta["camera_angle_x"]
        focal = 1.0
        self.width = self.config.resw 
        self.height = self.config.resh 

        for frame in tqdm(meta['frames']):    
            img_path = os.path.join(self.config.dataset_root(), frame['file_path'] + ".png")
            transform = np.array(frame['transform_matrix'])
            # view to world transform
            R = transform[:3, :3]
            T = transform[:3, 3]
            img = Image()
            img.load_from_file(img_path)
            if (self.config.white_bkgd):
                img.blend(np.ones(3))
            else:
                img.blend(np.zeros(3))

            T = T * self.config.scene_scale
            cam_info = CameraInfo(
                FovY=fov_x,
                R=R,
                T=T,
                ResW=img.W,
                ResH=img.H
            )
            img_info = img.info
            self._cam_img_pairs.append(
                CameraImagePair(cam_info, img_info))
    
        if (self.config.enable_rays):
            print("Generating Rays")
        self.N = len(self._cam_img_pairs)

    def get_point_cloud(self):
        ply_file_path = os.path.join(self.config.env_config.dataset_root, "nerf_blender_init.ply")
        pcd = fetchPly(ply_file_path)
        return pcd
    
def create_dataset(env_config, obj_name="lego", usage="train"):
    config = NeRFBlenderDatasetConfig(env_config)
    config.obj_name = obj_name
    config.usage = usage
    dataset = NeRFBlenderDataset(config)
    return dataset