from ...base import ProjectConfigBase, ProjectBase
from module.model.gaussian.vanilla import GaussianModel 

# renderer
# from app.renderer.gaussian_rasterizer.reprod import create_gaussian_renderer as create_reprod_renderer 
from app.diff_renderer.gaussian_rasterizer.vanilla import create_gaussian_renderer as create_vanilla_renderer 
# from app.renderer.gaussian_rasterizer.vanilla_ing import create_gaussian_renderer as create_vanilla_ing_renderer
from app.diff_renderer.gaussian_rasterizer.inno_reprod import create_gaussian_renderer as create_inno_reprod_renderer
# from app.renderer.gaussian_rasterizer.inno_zzh import create_gaussian_renderer as 
# from app.renderer.gaussian_rasterizer.inno_split import create_gaussian_renderer as create_inno_split_renderer

from module.utils.camera.basic import Camera
from module.utils.video.av import write_mp4
from app.pipeline.nvs.eval import NVSEvalPipelineConfig, NVSEvalPipeline 

from loguru import logger 
import os 
from typing import NamedTuple 
import numpy as np 
import torch 
import matplotlib.pyplot as plt

class EvalGaussianProjectConfig(ProjectConfigBase):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.name = "reproduce_gaussian_project"
        self.usage = "render"

class EvalGaussianProjectParams(NamedTuple):
    dataset_name: str 
    obj_name: str 
    ckpt_path: str
    output_name: str 
    render_name: str = "vanilla"
    benchmarks: list = ["psnr"] # "psnr" , "ssim", "lpips"

class EvalGaussianProject(ProjectBase):
    def __init__(self, config: EvalGaussianProjectConfig):
        super().__init__(config)
        # config, target_path, log
        self.model = GaussianModel(3)
        self.create_renderer = {
            # "inno_split": create_inno_split_renderer,
            # 'vanilla_ing': create_vanilla_ing_renderer,
            # 'reprod': create_reprod_renderer,
            "inno_reprod": create_inno_reprod_renderer,
            # "inno_zzh": create_inno_zzh_renderer
            'vanilla': create_vanilla_renderer
        }

    def run(self, params: EvalGaussianProjectParams):
        # load checkpoint
        self.model.load_ply(params.ckpt_path)
        result = None 
        if self.config.usage == "render":
            logger.info(f"Rendering from check point {params.ckpt_path} to {params.output_name}.png")
            self.render(params.output_name, params.dataset_name, params.render_name)
            result = True 
        elif self.config.usage == 'demo':
            logger.info(f"Rendering from check point {params.ckpt_path} to {params.output_name}.mp4")
            self.demo(params.output_name, params.dataset_name, params.render_name)
            result = True 
        elif self.config.usage == "eval":
            result = self.eval(
                params.dataset_name, 
                params.obj_name, 
                params.render_name, 
                params.benchmarks)

        return result

    def eval(self, dataset_name = "nerf_blender", obj_name ="lego", render_name="vanilla", benchmarks = ["psnr"]):
        pipeline_config = NVSEvalPipelineConfig(self.config.env_config)
        # configure pipeline 
        pipeline_config.proj_name = self.config.name
        pipeline_config.name = "nvs_eval_pipeline" + "_" + dataset_name + "_" + obj_name
        pipeline_config.dataset_name = dataset_name 
        pipeline_config.obj_name = obj_name 
        pipeline_config.metric_types = benchmarks
        # create and run pipeline
        pipeline = NVSEvalPipeline(pipeline_config)
        renderer = self.create_renderer[render_name](self.config.env_config)
        result = pipeline.run(self.model, renderer)
        return result


    def render(self, output_name, dataset_name="nerf_blender", render_name='vanilla'):
        # init renderer
        renderer = self.create_renderer[render_name](self.config.env_config)
        theta = 2 * np.pi / 180 * 0
        camera = Camera("FlipY")
        if (dataset_name == "nerf_blender"):
            # nerf 
            camera.lookat(2 * np.array([np.cos(theta), np.sin(theta), 1]), np.array([0, 0, 0]))
        else:
            # colmap
            camera.lookat(2 * np.array([np.cos(theta), -1, np.sin(theta)]), np.array([0, 0, 0]))
        camera.set_res(1600, 1600)

        img = renderer.render(camera, self.model)["render"]
        img_np = img.detach().cpu().clone()
        img_np=img_np.numpy().transpose(1, 2, 0).clip(0, 1)  
        # flip y & to uint
        img_np = img_np[::-1, :, :]
        plt.imsave(os.path.join(self.target_path, output_name + '.png'), img_np)

    def demo(self, output_name, dataset_name="nerf_blender", render_name='vanilla'):
        # init renderer
        img_list = []
        with torch.no_grad():
            for i in range(180):
                renderer = self.create_renderer[render_name](self.config.env_config)
                theta = 2 * np.pi / 180 * i 
                camera = Camera("FlipY")
                if (dataset_name == "nerf_blender"):
                    # nerf 
                    camera.lookat(2 * np.array([np.cos(theta), np.sin(theta), 1]), np.array([0, 0, 0]))
                else:
                    # colmap
                    camera.up = np.array([0, -1, 0])
                    camera.lookat(2 * np.array([np.cos(theta), -1, np.sin(theta)]), np.array([0, 0, 0]))
                camera.set_res(1600, 1600)

                img = renderer.render(camera, self.model)["render"]
                # if i % 4 == 0:
                # torchvision.utils.save_image(img, os.path.join(self.target_path, '{0:05d}'.format(i)+'.png'))
                img_np = img.detach().cpu().clone()
                img_np=img_np.numpy().transpose(1, 2, 0).clip(0, 1)
                # flip y & to uint
                img_np = img_np[::-1, :, :]
                img_list.append(img_np)
                torch.cuda.empty_cache()
        
        write_mp4(img_list, output_name, self.target_path, fps=30)