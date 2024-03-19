from ...base import ProjectConfigBase, ProjectBase
from loguru import logger 
import os 
from typing import NamedTuple 

from app.trainer.nvs.gs.basic import GaussianTrainerParams
# train and eval pipeline
from app.pipeline.nvs.gs.train import GaussianTrainPipelineConfig, GaussianTrainPipeline
from app.pipeline.nvs.eval import NVSEvalPipelineConfig, NVSEvalPipeline 
# model
from module.model.gaussian.vanilla import GaussianModel
# from module.model.vanilla_gaussian.model import GaussianModel
# renderer
from app.diff_renderer.gaussian_rasterizer.vanilla import create_gaussian_renderer as create_vanilla_renderer 
from app.diff_renderer.gaussian_rasterizer.inno_reprod import create_gaussian_renderer as create_inno_reprod_renderer
from app.diff_renderer.gaussian_rasterizer.inno_torch import create_gaussian_renderer as create_inno_torch_renderer

class TrainGaussianProjectConfig(ProjectConfigBase):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.name = "reimplement gaussian project"
        self.usage = "train"
        self.sh_deg = 3
       
class TrainGaussianProjectParams(NamedTuple):
    dataset_name: str = "nerf_blender"
    obj_name: str = "lego"
    init_scene: dict = {}
    trainer_name: str = "plain"
    train_params: GaussianTrainerParams = GaussianTrainerParams(),
    render_name: str = "vanilla"
    loss_name: str = "l1"
    metric_types: list = ["psnr", "ssim", "lpips"]

class TrainGaussianProject(ProjectBase):
    def __init__(self, config: TrainGaussianProjectConfig):
        super().__init__(config)
        
        self.create_renderer = {
            'vanilla': create_vanilla_renderer,
            # 'vanilla_ing': create_vanilla_ing_renderer,
            # 'reprod': create_reprod_renderer,
            "inno_reprod": create_inno_reprod_renderer,
            "inno_torch": create_inno_torch_renderer,
            # "reprod_noxyz": create_reprod_noxyz_renderer,
        }
    
    def run(self, params: TrainGaussianProjectParams):
        init_scene = params.init_scene
        self.model = GaussianModel(self.config.sh_deg)
        if init_scene["type"] == "ckpt":
            ckpt_path = os.path.join(
                self.config.env_config.pretrained_path,
                init_scene["ckpt_path"],
                "_".join([init_scene["dataset_name"],
                    init_scene["obj_name"],
                    init_scene["postfix"]]) + ".ply")
            if not os.path.exists(ckpt_path):
                logger.error(f"ckpt {ckpt_path} not exists, use random by default")
                ckpt_path = os.path.join(self.config.env_config.pretrained_path, "random.ply")
            else:
                logger.info(f"loading ckpt from {ckpt_path}")
            self.model.load_ply(ckpt_path)

        # init render
        renderer = self.create_renderer[params.render_name](self.config.env_config)
        train_config = GaussianTrainPipelineConfig(self.config.env_config)
        train_config.proj_name = self.config.proj_name
        
        train_config.name = f"{params.dataset_name}_{params.obj_name}_{params.trainer_name}_{params.loss_name}_{init_scene['postfix']}_train_pipeline"
        train_config.dataset_name = params.dataset_name 
        train_config.obj_name = params.obj_name
        train_config.trainer_name = params.trainer_name 
        train_config.loss_name = params.loss_name 
        train_pipeline = GaussianTrainPipeline(train_config)

        # if eval, load ckpt
        eval_config = NVSEvalPipelineConfig(self.config.env_config)
        if self.config.usage == "eval":
            logger.info(f"evaluating on {params.dataset_name} {params.obj_name} with {params.render_name} renderer, {params.trainer_name} strategy and {params.loss_name} loss")
            save_iter = params.train_params.max_iterations
            ckpt_path = os.path.join(train_pipeline.target_path, "_".join([params.train_params.name, str(save_iter)]) + ".ply")
            self.model.load_ply(ckpt_path)

        # run train_pipeline
        if self.config.usage == "train":
            logger.info(f"training on {params.dataset_name} {params.obj_name} with {params.render_name} renderer, {params.trainer_name} strategy, {params.loss_name} loss and {params.train_params.name} params")
            train_pipeline.run(self.model, renderer, params.train_params)
            logger.info("Train Finished, start Eval")
        
        # eval after train
        eval_config.dataset_name = params.dataset_name
        eval_config.obj_name = params.obj_name
        eval_config.proj_name = self.config.proj_name
        eval_config.metric_types = params.metric_types
        eval_config.name = f"{params.dataset_name}_{params.obj_name}_{params.trainer_name}_{params.loss_name}_{init_scene['postfix']}_eval_pipeline"
        eval_config.output_name = f"{params.dataset_name}_{params.obj_name}_{params.trainer_name}_{params.loss_name}_{params.train_params.name}"
        eval_pipeline = NVSEvalPipeline(eval_config)
        result = eval_pipeline.run(self.model, renderer)

        return result 
