from ...base import TrainerConfigBase, TrainProcessLogBase, TrainerBase
from module.utils.camera.basic import Camera

import torch
from tqdm import tqdm
from random import randint
import os 
from loguru import logger 
import matplotlib.pyplot as plt 

from dataclasses import dataclass

class GaussianTrainerConfig(TrainerConfigBase):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.optimzer: str = "adamw"
        self.model_name: str = "vanilla_gaussian"
        self.target_path = ""

@dataclass
class GaussianTrainerParams:
    name: str = "dummy_params"
    percent_dense = 0.01
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30000
    pano_lr = 0.1
    pano_h = 128
    opacity_lr = 0.05
    scaling_lr = 0.005
    feature_lr = 0.0025
    rotation_lr = 0.001
    saving_iterations = [7000, 30000]
    max_iterations = 30000

class GaussianTrainer:
    def __init__(self,
        config: GaussianTrainerConfig):
        self.config = config

    def train(self, gaussians, pano, dataset, renderer, loss_fn, params: GaussianTrainerParams):
        iterations = params.max_iterations 
        first_iter = 0
        progress_bar = tqdm(range(first_iter, iterations), desc="Training Progress")
        first_iter += 1
        pairs = None 
        logger.info(f"Training with {params.name}")
        # train steup
        gaussians.training_setup(params)
        pano.requires_grad = True
        pano_optimizer = torch.optim.AdamW([pano], lr=params.pano_lr)

        for iteration in range(first_iter, iterations + 1):
            gaussians.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
            # pick a random camera
            if not pairs or len(pairs) == 0:
                pairs = dataset.pairs()
            pair = pairs.pop(randint(0, len(pairs)-1))
            
            camera = Camera()
            camera.from_info(pair.cam) # flip z
            camera.flip() # to flip y
            render_pkg = renderer.render(camera, gaussians, pano)
            image = render_pkg["render"]
            
            gt_image = torch.tensor(pair.img.data.transpose(2, 0, 1)).float().cuda()
            
            loss = loss_fn(image, gt_image)
            loss.backward()
            
            with torch.no_grad():
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                    progress_bar.update(10)
                
                if iteration == iterations:
                    progress_bar.close()                
                # save
                if (iteration in params.saving_iterations):
                    logger.info(f"\n[ITER {iteration}] Saving Gaussians")
                    # scene.save(iteration)
                    point_cloud_name = f"{params.name}_{iteration}.ply"
                    gaussians.save_ply(os.path.join(self.config.target_path, point_cloud_name))
                    pano_img_np = pano.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
                    # save
                    plt.imsave(os.path.join(self.config.target_path, f"pano_{iteration}.png"), pano_img_np)

                # densification

                # optimize
                if iteration < iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    pano_optimizer.step()
                    pano_optimizer.zero_grad(set_to_none=True)

def create_trainer(env_config, target_path):
    trainer_config = GaussianTrainerConfig(env_config)
    trainer_config.target_path = target_path
    return GaussianTrainer(trainer_config)
