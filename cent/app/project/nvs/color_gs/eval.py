from ...base import ProjectConfigBase, ProjectBase
from module.model.gaussian.color import GaussianModel 

# renderer
# from app.diff_renderer.gaussian_rasterizer.vanilla_reprod import create_gaussian_renderer as create_vanilla_reprod_renderer 
from app.diff_renderer.gaussian_rasterizer.vanilla import create_gaussian_renderer as create_vanilla_renderer 
from app.diff_renderer.gaussian_rasterizer.inno_reprod import create_gaussian_renderer as create_inno_reprod_renderer
from app.diff_renderer.gaussian_rasterizer.inno_split import create_gaussian_renderer as create_inno_split_renderer
from app.diff_renderer.gaussian_rasterizer.inno_torch import create_gaussian_renderer as create_inno_torch_renderer
from app.diff_renderer.gaussian_rasterizer.split import create_gaussian_renderer as create_split_renderer

# scene
from module.data.point_cloud import sphere_point_cloud
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
    scene: dict
    output_name: str 
    render_name: str = "vanilla"
    benchmarks: list = ["psnr"] # "psnr" , "ssim", "lpips"

class EvalGaussianProject(ProjectBase):
    def __init__(self, config: EvalGaussianProjectConfig):
        super().__init__(config)
        # config, target_path, log
        self.model = GaussianModel()
        self.create_renderer = {
            # 'vanilla_ing': create_vanilla_ing_renderer,
            # 'reprod': create_reprod_renderer,
            "inno_reprod": create_inno_reprod_renderer,
            "inno_split": create_inno_split_renderer,
            "inno_torch": create_inno_torch_renderer,
            "split": create_split_renderer,
            'vanilla': create_vanilla_renderer
        }

    def run(self, params: EvalGaussianProjectParams):
        self.params = params
        result = None 
        if self.config.usage == "render":
            self.render()
            result = True 
        elif self.config.usage == 'demo':
            self.demo()
            result = True 
        elif self.config.usage == "eval":
            assert params.scene["type"] == "ckpt"
            result = self.eval()

        return result

    def eval(self):
        scene = self.params.scene
        assert scene["type"] == "ckpt"
        self.model.load_ply(scene["ckpt_path"])

        pipeline_config = NVSEvalPipelineConfig(self.config.env_config)
        # configure pipeline 
        pipeline_config.proj_name = self.config.name
        pipeline_config.name = "nvs_eval_pipeline" + "_" + scene["dataset_name"] + "_" + scene["obj_name"]
        pipeline_config.dataset_name = scene["dataset_name"]
        pipeline_config.obj_name = scene["obj_name"] 
        pipeline_config.metric_types = self.params.benchmarks
        # create and run pipeline
        pipeline = NVSEvalPipeline(pipeline_config)
        
        renderer = self.create_renderer[self.params.render_name](self.config.env_config)
        result = pipeline.run(self.model, renderer)
        return result

    def render(self):
        params = self.params
        theta = 2 * np.pi / 180 * 0
        camera = Camera("FlipY")
        camera.lookat(3 * np.array([np.cos(theta), np.sin(theta), 1]), np.array([0, 0, 0]))
        if params.scene["type"] == "ckpt":
            ckpt = params.scene["ckpt_path"]
            self.model.load_ply(ckpt)
            if (params.scene["dataset_name"] == "mip360"):
                # colmap
                camera.lookat(2 * np.array([np.cos(theta), -1, np.sin(theta)]), np.array([0, 0, 0]))

        elif params.scene["type"] == "sphere":
            r = params.scene["r"]
            N = params.scene["N"]
            pc = sphere_point_cloud(r, N)
            self.model.create_from_pcd(pc, r)
        
        # init renderer
        renderer = self.create_renderer[params.render_name](self.config.env_config)

        camera.set_res(1600, 1600)
        # camera.set_res(1024, 1024)
        # camera.set_res(1024, 256)
        # camera.set_res(256, 256)
        # camera.set_res(3200, 3200)
        img = renderer.render(camera, self.model)["render"]
        img_np = img.detach().cpu().clone()
        img_np=img_np.numpy().transpose(1, 2, 0).clip(0, 1)  
        # flip y & to uint
        img_np = img_np[::-1, :, :]
        plt.imsave(os.path.join(self.target_path, params.output_name + '.png'), img_np)

    def demo(self):
        # init renderer
        params = self.params
        img_list = []
        renderer = self.create_renderer[params.render_name](self.config.env_config)
        camera = Camera("FlipY")
         
        if params.scene["type"] == "ckpt":
            ckpt = params.scene["ckpt_path"]
            self.model.load_ply(ckpt)

        elif params.scene["type"] == "sphere":
            r = params.scene["r"]
            N = params.scene["N"]
            pc = sphere_point_cloud(r, N)
            self.model.create_from_pcd(pc, r)

        with torch.no_grad():
            for i in range(180):
                theta = 2 * np.pi / 180 * i 
                camera.set_res(1600, 1600)
                camera.lookat(2 * np.array([np.cos(theta), np.sin(theta), 1]), np.array([0, 0, 0]))
                if (params.scene["dataset_name"] == "mip360"):
                    # colmap
                    camera.up = np.array([0, -1, 0])
                    camera.lookat(2 * np.array([np.cos(theta), -1, np.sin(theta)]), np.array([0, 0, 0]))

                img = renderer.render(camera, self.model)["render"]
                # if i % 4 == 0:
                # torchvision.utils.save_image(img, os.path.join(self.target_path, '{0:05d}'.format(i)+'.png'))
                img_np = img.detach().cpu().clone()
                img_np=img_np.numpy().transpose(1, 2, 0).clip(0, 1)
                # flip y & to uint
                img_np = img_np[::-1, :, :]
                img_list.append(img_np)
                torch.cuda.empty_cache()
        
        write_mp4(img_list, params.output_name, self.target_path, fps=30)