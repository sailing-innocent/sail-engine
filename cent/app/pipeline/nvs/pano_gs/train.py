from ..base import NVSPipelineConfig, NVSPipeline

# dataset support 
from module.dataset.nvs.blender.dataset import create_dataset as create_nerf_blender_dataset
from module.dataset.nvs.mip360.dataset import create_dataset as create_mip360_dataset
from module.dataset.nvs.tank_temple.dataset import create_dataset as create_tank_temple_dataset

# trainer 
from app.trainer.nvs.pano_gs.basic import create_trainer as create_basic_trainer
from app.trainer.nvs.pano_gs.vanilla import create_trainer as create_vanilla_trainer
# loss
from lib.reimpl.vanilla_diff_gaussian.utils.loss_utils import l1_loss, ssim
from loguru import logger 

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
            "vanilla": create_vanilla_trainer
        }
        logger.info(f"setup trainer with {config.trainer_name}")
        # loss
        lambda_dssim = config.lambda_dssim
        self.create_loss = {
            "l1": l1_loss,
            "l1+ssim": lambda target, gt: (1 - lambda_dssim) * l1_loss(target, gt) + lambda_dssim * (1 - ssim(target, gt))
        }

    def run(self, model, pano, renderer, train_params):
        dataset = self.create_dataset[self.config.dataset_name](self.config.env_config, self.config.obj_name, "train")
        loss_fn = self.create_loss[self.config.loss_name]
        trainer = self.create_trainer[self.config.trainer_name](self.config.env_config, self.target_path)
        trainer.train(model, pano, dataset, renderer, loss_fn, train_params)
