from ...base import TrainerConfigBase, TrainProcessLogBase, TrainerBase
from .vanilla import GaussianVanillaTrainerParams
from module.utils.camera.basic import Camera
import torch
from tqdm import tqdm
from random import randint
import os 
from loguru import logger 

from module.model.gaussian.model import Gaussians2D, Gaussians2DTrainArgs
from app.diff_renderer.gaussian_sampler.inno import GaussianSampler
from app.diff_renderer.gaussian_projector.inno import GaussianProjector

import matplotlib.pyplot as plt

class GaussianTrainerConfig(TrainerConfigBase):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.optimzer: str = "adamw"
        self.target_path = ""

class GaussianTrainerProcessLog(TrainProcessLogBase):
    def __init__(self):
        super().__init__()

    def load(self):
        pass 

    def save(self):
        pass 

class GaussianEpipolarTrainerParams(GaussianVanillaTrainerParams):
    def __init__(self):
        super().__init__()
        self.name = "epipolar params"
        self.opacity_reset_interval = 3000
        self.epipolar_interval = 1777
        self.densify_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

def show_compare_img(img1, img2, path=None):
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    if path is not None:
        plt.savefig(path)

def epipolar_update(gs, pairs, N_sparse,save_path):
    logger.info("Epipolar Update")
    sampler = GaussianSampler()
    projector = GaussianProjector()
    N_batch = N_sparse
    N_ITER = 4
    N_SHOW = 2
    N_SUB_ITER = 200
    N_SUB_LOG = 250
    cam_infos = []
    cams = []
    imgs = []
    target_imgs = []    
    gs2ds = []
    for idx in range(N_ITER):
        for i in range(N_batch):
            cam_infos.append(pairs[i].cam)
            imgs.append(pairs[i].img)
            cam = Camera("FlipY")
            cam.from_info(cam_infos[i])
            cam.set_res(imgs[i].W, imgs[i].H)
            cams.append(cam)
            target_img = torch.tensor(imgs[i].data.transpose(2, 0, 1), dtype=torch.float32).cuda()
            target_img._requires_grad = False
            target_imgs.append(target_img)

        train_args = Gaussians2DTrainArgs()
        # iterate each view
        for i in range(N_batch):
            gs2d = projector.project(gs, cams[i])
            gs2d.clone_detach()
            gs2d.requires_grad()
            gs2d.training_setup(train_args)
            for sub_i in range(N_SUB_ITER):
                result_img = sampler.sample(gs2d, imgs[i].W, imgs[i].H, cams[i].info.FovY)
                loss = torch.functional.F.mse_loss(target_imgs[i], result_img)
                loss.backward(retain_graph=True)
                with torch.no_grad():
                    gs2d.optim.step()
                    gs2d.optim.zero_grad()
                    if (sub_i+1) % N_SUB_LOG == 0:
                        print(f'Iter {sub_i}, loss: {loss.item()}')
                        result_img_np = result_img.detach().cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)
                        # compare
                        plt.subplot(1, 2, 1)
                        plt.imshow(imgs[i].data)
                        plt.subplot(1, 2, 2)
                        plt.imshow(result_img_np)
                        plt.show()
            gs2ds.append(gs2d)

        with torch.no_grad():
            if (idx + 1) % N_SHOW == 0:
                for i in range(N_batch):
                    gs2d = projector.project(gs, cams[i])
                    gs2d.clone_detach()
                    result_img = sampler.sample(gs2d, imgs[i].W, imgs[i].H, cams[i].info.FovY)
                    result_img_np = result_img.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
                    show_compare_img(imgs[i].data, result_img_np, f"{save_path}/{idx}_{i}.png")

            print("epipolar view")
            gs.epipolar_view(gs2ds, cams)

class GaussianTrainer(TrainerBase):
    """
    inherited
        - config
        - eval_results
        - train_process_logs
        - weights
    """
    def __init__(self,
        config: GaussianTrainerConfig):
        super().__init__(config)

    def train(self, gaussians, dataset, renderer, loss_fn, params: GaussianVanillaTrainerParams):
        process_log = GaussianTrainerProcessLog()
        iterations = params.max_iterations
        first_iter = 0
        progress_bar = tqdm(range(first_iter, iterations), desc="Training Progress")
        first_iter += 1
        pairs = None 

        logger.info(f"Training with {params.name}")
        # train steup
        gaussians.training_setup(params)

        for iteration in range(first_iter, iterations + 1):
            gaussians.update_learning_rate(iteration)

            # every 1000 iter we increase the level of SH degrees
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
    
            # pick a random camera
            if not pairs or len(pairs) == 0:
                pairs = dataset.pairs(params.data_limit, params.data_shuffle)
            pair = pairs.pop(randint(0, len(pairs)-1))
            
            # render
            if dataset.name == "nerf_blender" or dataset.name == "tank_temple":
                camera = Camera()
                camera.from_info(pair.cam) # flip z
                camera.flip() # to flip y
            elif dataset.name == "mip360":
                camera = Camera("FlipY")
                camera.from_info(pair.cam)
            else:
                raise NotImplementedError
            
            render_pkg = renderer.render(camera, gaussians)
            image, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
            # load ground truth
            gt_image = torch.tensor(pair.img.data.transpose(2, 0, 1)).float().cuda()
            
            loss = loss_fn(image, gt_image)
            loss.backward()
            
            if iteration % params.epipolar_interval == 0:
                pairs = dataset.pairs(params.data_limit, params.data_shuffle)
                epipolar_update(gaussians, pairs, params.data_limit, self.config.target_path)

            else:
                with torch.no_grad():
                    # update and optim
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

                    # densification
                    if iteration < params.densify_until_iter:
                        # Keep Track of max radii in image-space for pruning
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if iteration > params.densify_from_iter and iteration % params.densify_interval == 0:
                            size_threshold = 20 if iteration > params.opacity_reset_interval else None
                            gaussians.densify_and_prune(params.densify_grad_threshold, 0.005, 1, size_threshold)

                        # todo white
                        if iteration % params.opacity_reset_interval == 0 or (iteration == params.densify_from_iter):
                            gaussians.reset_opacity()

                    # optimize
                    if iteration < iterations:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)

        return process_log

def create_trainer(env_config, target_path):
    trainer_config = GaussianTrainerConfig(env_config)
    trainer_config.target_path = target_path
    return GaussianTrainer(trainer_config)
