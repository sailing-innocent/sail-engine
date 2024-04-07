
from ..base import NVSDatasetConfig, NVSDataset 
from ..base import CameraImagePair, RayColorPair
from module.utils.image.basic import Image
from module.data.image_info import ImageInfo
from module.data.camera_info import CameraInfo

import sys 
import os 
import numpy as np 
import torch 
from PIL import Image as PILImage

from loguru import logger  
from module.utils.np.graphics import focal2fov
from module.utils.np.transform import qvec2R
from .util import read_extrinsics_text, read_intrinsics_text, read_extrinsics_binary, read_intrinsics_binary
from .util import read_points3D_binary, read_points3D_text
# utilities
from module.utils.pointcloud.io import fetchPly, storePly
from module.tool.mono_depth_predictor.unidepth import MonoDepthPredictor

class ColmapDatasetConfig(NVSDatasetConfig):
    """
    Inherited
        - env_config
        - batch_size = 64
        - usage = "train"
    """
    def __init__(self, env_config):
        super().__init__(env_config)
        self.name = "colmap"
        self.obj_name = "dummy"
        self.obj_list = ['dummy']
        self.images = "images_4"
        self.type = "RGB"
        self.llffhold = 8

    def dataset_root(self):
        return os.path.join(self.env_config.dataset_root, "colmap", self.obj_name)

class ColmapDataset(NVSDataset):
    """
    Inherited
        - config
    """ 
    def __init__(self, config: ColmapDatasetConfig):
        super().__init__(config)
        self.depth_predictor = MonoDepthPredictor()
        
    def _load_dataset(self):
        dataset_path = self.config.dataset_root()
        cameras_extrinsic_file = os.path.join(dataset_path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(dataset_path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        reading_dir = "images" if self.config.images == None else self.config.images
        self._read_colmap(cam_extrinsics, cam_intrinsics, os.path.join(dataset_path, reading_dir))

    def get_point_cloud(self):
        path = self.config.dataset_root()
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            logger.info("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
                
        return pcd

    # the scene itself is modeled in flip-y coordinate system
    def _read_colmap(self, cam_extrinsics, cam_intrinsics, images_folder):
        for idx, key in enumerate(cam_extrinsics):
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
            sys.stdout.flush()
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width

            uid = intr.id
            R = qvec2R(extr.qvec)
            T = np.array(extr.tvec)
            # world to view translation need to be inverted
            T = -np.matmul(R.T, T)

            R = np.transpose(R)

            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            img = Image()
            img.load_from_file(image_path)
            img.to_float32()
            img.flip_y() # for flip-y coordinate system


            cam_info = CameraInfo(
                FovY = FovY,
                FovX = FovX,
                R = R,
                T = T,
                ResW = img.W,
                ResH = img.H
            )

            if self.config.type == "RGBD":
                depth_folder = images_folder.replace("images", "depth")
                os.makedirs(depth_folder, exist_ok=True)
                depth_path = os.path.join(depth_folder, os.path.basename(extr.name).replace("JPG", "npy"))
                if not os.path.exists(depth_path):
                    # prepropose
                    img_torch = torch.from_numpy(img.data).permute(2, 0, 1).unsqueeze(0).float().cuda()
                    intr_torch = torch.from_numpy(cam_info.K).float().cuda()
                    depth = self.depth_predictor.predict(img_torch, intr_torch)
                    depth_np = depth.detach().cpu().numpy()[:, :, np.newaxis] # H W 1
                    np.save(depth_path, depth_np)

                depth_np = np.load(depth_path)
                img.merge_data(depth_np)

            img_info = img.info
            # the colmap images is FlipY coordinate
            self._cam_img_pairs.append(
                CameraImagePair(cam_info, img_info))
        sys.stdout.write("\n")

        self.height = height
        self.width = width
        self.N = len(self._cam_img_pairs)