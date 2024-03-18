from ..base import NVSDatasetConfig, NVSDataset 
from ..base import CameraImagePair, RayColorPair

from module.utils.image.basic import Image
from module.data.image_info import ImageInfo
from module.data.camera_info import CameraInfo

import os 
from tqdm import tqdm 
import numpy as np  
from loguru import logger 

from module.utils.pointcloud.io import fetchPly, storePly


class NSVFDatasetConfig(NVSDatasetConfig):
    """
    Inherited
        - env_config
        - batch_size = 64
        - usage = "train"
    """
    def __init__(self, env_config):
        super().__init__(env_config)
        self.name = "nsvf"

    def dataset_root(self):
        return os.path.join(self.env_config.dataset_root, "nsvf", self.obj_name)

class NSVFDataset(NVSDataset):
    """
    Inherited
        - config
    """ 
    def __init__(self, config: NSVFDatasetConfig):
        super().__init__(config)

    def get_point_cloud(self):
        ply_file_path = os.path.join(self.config.dataset_root(), "pointcloud_norm.ply")
        pcd = fetchPly(ply_file_path)
        return pcd

    def _load_dataset(self):
        # load dataset
        dataset_path = self.config.dataset_root()
        assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist."
        
        # TODO: scene scale
        # permutation
        # epoch size

        self.split = "train"
        split_name = "train"
        logger.info(f"Reading {split_name} split of NSVF dataset {self.config.obj_name} from {dataset_path}")
        
        def sort_key(x):
            if len(x) > 2 and x[1] == "_":
                return x[2:]
            return x
        def look_for_dir(cands, required=True):
            for cand in cands:
                if os.path.isdir(os.path.join(dataset_path, cand)):
                    return cand
            if required:
                assert False, "None of " + str(cands) + " found in data directory"
            return ""

        img_dir_name = look_for_dir(["images", "image", "rgb"])
        pose_dir_name = look_for_dir(["poses", "pose"])

        orig_img_files = sorted(
            os.listdir(
                os.path.join(dataset_path, img_dir_name)), key=sort_key)

        # print(orig_img_files) # ['0_00001.png', ...]
        # split train|test|val
        if self.split == "train" or self.split == "test_train":
            img_files = [x for x in orig_img_files if x.startswith("0_")]
        elif self.split == "val":
            img_files = [x for x in orig_img_files if x.startswith("1_")]
        elif self.split == "test":
            test_img_files = [x for x in orig_img_files if x.startswith("2_")]
            if len(test_img_files) == 0:
                test_img_files = [x for x in orig_img_files if x.startswith("1_")]
            img_files = test_img_files
        else:
            img_files = orig_img_files
        
        if len(img_files) == 0:
            if self.split == "train":
                img_files = [x for i, x in enumerate(orig_img_files) if i % 16 != 0]
            else:
                img_files = orig_img_files[::16]

        assert len(img_files) > 0, "No matching images in directory: " + os.path.join(root, img_dir_name)
        self.img_files = img_files

        intrin_path = os.path.join(dataset_path, "intrinsics.txt")
        assert os.path.exists(intrin_path), "intrinsics unavailable"
        try:
            K: np.ndarray = np.loadtxt(intrin_path)
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
        except:
            # Weird format sometimes in NSVF data
            with open(intrin_path, "r") as f:
                spl = f.readline().split()
                fx = fy = float(spl[0])
                cx = float(spl[1])
                cy = float(spl[2])

        fov_y = 2 * np.arctan2(cy, fy)

        for img_fname in tqdm(img_files):
            img_path = os.path.join(dataset_path, img_dir_name, img_fname)
            img = Image()
            img.load_from_file(img_path)
            pose_fname = os.path.splitext(img_fname)[0] + ".txt"
            pose_path = os.path.join(dataset_path, pose_dir_name, pose_fname)

            # logger.info(f"Reading {img_path} and {pose_path}")
            cam_mtx = np.loadtxt(pose_path).reshape(-1, 4)
            
            cam_mtx = np.loadtxt(pose_path).reshape(-1, 4)
            if len(cam_mtx) == 3:
                bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
                cam_mtx = np.concatenate([cam_mtx, bottom], axis=0)
            
            R = cam_mtx[:3, :3]
            T = cam_mtx[:3, 3]

            # TODO: scale the scene and modify T
            cam_info = CameraInfo(
                FovY=fov_y, # for aspect = 1
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