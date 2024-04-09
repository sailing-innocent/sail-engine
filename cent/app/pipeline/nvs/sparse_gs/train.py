from ..base import NVSPipelineConfig, NVSPipeline

# dataset support 
from module.dataset.nvs.blender.dataset import create_dataset as create_nerf_blender_dataset
from module.dataset.nvs.mip360.dataset import create_dataset as create_mip360_dataset
from module.dataset.nvs.tank_temple.dataset import create_dataset as create_tank_temple_dataset

# trainer 
from app.trainer.nvs.sparse_gs.basic import create_trainer as create_basic_trainer
from app.trainer.nvs.sparse_gs.vanilla import create_trainer as create_vanilla_trainer
from app.trainer.nvs.sparse_gs.epipolar import create_trainer as create_epipolar_trainer
# loss
from lib.reimpl.vanilla_diff_gaussian.utils.loss_utils import l1_loss, ssim
from loguru import logger 
import os 
import json
import matplotlib.pyplot as plt 
import numpy as np 

class GaussianTrainPipelineConfig(NVSPipelineConfig):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.name: str = "gaussian_train_pipeline"
        self.dataset_name: str = "nerf_blender"
        self.obj_name: str = "lego"
        self.output_name: str = "nerf_blender_lego"
        self.trainer_name: str = "basic"
        self.loss_name = "l1"
        self.lambda_dssim: float = 0.2 

class GaussianTrainPipeline(NVSPipeline):
    def __init__(self, config: GaussianTrainPipelineConfig):
        super().__init__(config)
        # load dataset
        self.create_dataset = {
            "nerf_blender": create_nerf_blender_dataset,
            "mip360": create_mip360_dataset,
            "tank_temple": create_tank_temple_dataset
        }
        # trainer
        self.create_trainer = {
            "basic": create_basic_trainer,
            "vanilla": create_vanilla_trainer,
            "epipolar": create_epipolar_trainer
        }
        logger.info(f"setup trainer with {config.trainer_name}")
        # loss
        lambda_dssim = config.lambda_dssim
        self.create_loss = {
            "l1": l1_loss,
            "l1+ssim": lambda target, gt: (1 - lambda_dssim) * l1_loss(target, gt) + lambda_dssim * (1 - ssim(target, gt))
        }

    def run(self, model, renderer, train_params):
        dataset = self.create_dataset[self.config.dataset_name](self.config.env_config, self.config.obj_name, "train")
        loss_fn = self.create_loss[self.config.loss_name]
        trainer = self.create_trainer[self.config.trainer_name](self.config.env_config, self.target_path)
        if len(model.get_xyz) == 0:
            init_pcd = dataset.get_point_cloud()
            model.create_from_pcd(init_pcd, 1.0) 
        # model.create_from_pcd(init_pcd, 1.0) 
        trainer.train(model, dataset, renderer, loss_fn, train_params)
        # save pairs
        pairs = dataset.pairs(train_params.data_limit, train_params.data_shuffle)
        
        save_dir = os.path.join(self.target_path, "train_pairs")
        os.makedirs(save_dir, exist_ok=True)
        for idx, pair in enumerate(pairs):
            json_f = os.path.join(save_dir, f"{str(idx)}.json")
            # dump cam info
            with open(json_f, "w") as f:
                json.dump(pair.cam.to_dict(), f)
            # save img
            img_f = os.path.join(save_dir, f"{str(idx)}.png")
            plt.imsave(img_f, (pair.img.data * 255.0).astype(np.uint8))

