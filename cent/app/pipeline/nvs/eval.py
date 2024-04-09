from .base import NVSPipelineConfig, NVSPipeline
from app.evaluator.nvs import NVSEvaluator, NVSEvalParams

# dataset support
from module.dataset.nvs.blender.dataset import create_dataset as create_nerf_blender_dataset
from module.dataset.nvs.mip360.dataset import create_dataset as create_mip360_dataset
from module.dataset.nvs.tank_temple.dataset import create_dataset as create_tank_temple_dataset

import os 
from loguru import logger 

class NVSEvalPipelineConfig(NVSPipelineConfig):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.name = "gaussian_eval_pipeline"
        self.dataset_name = "nerf_blender"
        self.obj_name = "lego"
        self.ply_file = ""
        self.metric_types = ["psnr", "ssim", "lpips"]
        self.output_name = "nerf_blender_lego"

class NVSEvalPipeline(NVSPipeline):
    def __init__(self, config: NVSEvalPipelineConfig):
        super().__init__(config) 
        # load dataset 
        create_dataset = {
            "nerf_blender": create_nerf_blender_dataset,
            "mip360": create_mip360_dataset,
            "mip360d": create_mip360_dataset, # for eval only RGB
            "tank_temple": create_tank_temple_dataset
        }
        self.dataset = create_dataset[config.dataset_name](config.env_config, config.obj_name, "test")
        self.evaluator = NVSEvaluator()

    def run(self, model, renderer):
        params = NVSEvalParams(
            metric_types = self.config.metric_types,
            save = True,
            save_dir = os.path.join(self.target_path, self.config.output_name)
        )
        result = self.evaluator.eval(self.dataset, model, renderer, params)
        logger.info("eval result: ", result)
        return result 