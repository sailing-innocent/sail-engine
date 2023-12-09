from ..base import VisualizerConfigBase, VisualizerBase
from module.blender.util.wm import bopen, bclose
from module.blender.vis.view import vis_view
import numpy as np 
import cv2 as cv 
import os 

class MultiViewBlenderVisualizerConfig(VisualizerConfigBase):
    """
        - self.env_config 
    """
    def __init__(self, env_config):
        super().__init__(env_config)
        self.target_path = "multi_view"
        os.makedirs(os.path.join(self.env_config.blender_root, self.target_path), exist_ok=True)
        self.mainfile_name = "vis"
    
    @property
    def mainfile_path(self):
        return os.path.join(self.env_config.blender_root, self.target_path, self.mainfile_name + ".blend")

class MultiViewBlenderVisualizer(VisualizerBase):
    """
        self.config
    """
    def __init__(self, config: MultiViewBlenderVisualizerConfig):
        super().__init__(config)

    def visualize(self, dataset):
        mainfile_path = self.config.mainfile_path
        bopen(mainfile_path)
        # use image and camera pairs
        assert dataset.use_rays == False 

        for idx, (cam_info, img_info) in enumerate(dataset):
            img_src = img_info.data
            if img_info.ydown:
                img_src = img_src[::-1, :, :]
            # if the img is too large, resize it to 400 and keep aspect
            if img_src.shape[0] > 400:
                img_src = cv.resize(img_src, (400, int(400 * img_src.shape[0] / img_src.shape[1])))
            c2w = np.eye(4) 
            c2w[:3, :3] = cam_info.R
            c2w[:3, 3] = cam_info.T
            vis_view(img_src, c2w, "img_{}".format(idx))
        bclose(mainfile_path)
