from ..base import NVSPipelineConfig, NVSPipeline

# dataset support 
from module.dataset.nvs.blender.dataset import create_dataset as create_nerf_blender_dataset
from module.dataset.nvs.mip360.dataset import create_dataset as create_mip360_dataset
from module.dataset.nvs.tank_temple.dataset import create_dataset as create_tank_temple_dataset

# trainer 

from app.trainer.nvs.gaussian.plain import GaussianTrainerParams
## plain
from app.trainer.nvs.gaussian.plain import create_trainer as create_plain_trainer
## vanilla
from app.trainer.nvs.gaussian.vanilla import create_trainer as create_vanilla_trainer
from app.trainer.nvs.gaussian.vanilla import VanillaTrainerParams 
# ## panorama
# from app.trainer.nvs.gaussian.panorama import create_trainer as create_panorama_trainer
# from app.trainer.nvs.gaussian.panorama import GaussianTrainerParams as PanoramaTrainerParams 
# # pano
# from app.trainer.nvs.gaussian.pano import create_trainer as create_pano_trainer
# from app.trainer.nvs.gaussian.pano import GaussianTrainerParams as PanoTrainerParams 

# loss
from lib.reimpl.vanilla_diff_gaussian.utils.loss_utils import l1_loss, ssim

from loguru import logger 
import numpy as np 
import os 
import matplotlib.pyplot as plt 

class GaussianTrainPipelineConfig(NVSPipelineConfig):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.name: str = "gaussian_train_pipeline"
        self.dataset_name: str = "nerf_blender"
        self.obj_name: str = "lego"
        self.output_name: str = "nerf_blender_lego"
        self.trainer_name: str = "plain"
        self.loss_name = "l1"
        self.lambda_dssim: float = 0.2 

class GaussianTrainPipeline(NVSPipeline):
    def __init__(self, config: GaussianTrainPipelineConfig):
        super().__init__(config)
        # load dataset
        create_dataset = {
            "nerf_blender": create_nerf_blender_dataset,
            "mip360": create_mip360_dataset,
            "tank_temple": create_tank_temple_dataset
        }

        self.dataset = create_dataset[config.dataset_name](config.env_config, config.obj_name, "train")
        # init_pcd
        self.init_pcd = self.dataset.get_point_cloud()
        # trainer

        self.create_trainer = {
            "plain": create_plain_trainer,
            "vanilla": create_vanilla_trainer,
            # "panorama": create_panorama_trainer,
            # "pano": create_pano_trainer
        }
        logger.info(f"setup trainer with {config.trainer_name}")

        self.trainer = self.create_trainer[config.trainer_name](self.config.env_config, self.target_path)
        # loss
        lambda_dssim = config.lambda_dssim
        self.create_loss = {
            "l1": l1_loss,
            "l1+ssim": lambda target, gt: (1 - lambda_dssim) * l1_loss(target, gt) + lambda_dssim * (1 - ssim(target, gt))
        }
        self.loss_fn = self.create_loss[config.loss_name]

    def run(self, model, renderer, train_params):
        # init model
        model.create_from_pcd(self.init_pcd, 1.0) 
        self.trainer.train(model, self.dataset, renderer, self.loss_fn, train_params)
