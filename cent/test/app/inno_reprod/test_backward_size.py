from module.data.point_cloud import sphere_point_cloud
from module.utils.camera.basic import Camera
from module.model.gaussian.vanilla import GaussianModel
import numpy as np 
import torch 
import pytest 
from app.diff_renderer.gaussian_rasterizer.inno_reprod import create_gaussian_renderer as create_inno_reprod_renderer
from app.diff_renderer.gaussian_rasterizer.vanilla import create_gaussian_renderer as create_vanilla_renderer 
from mission.config.env import get_env_config
import matplotlib.pyplot as plt
from dataclasses import dataclass

def compare_show(img_1, img_2):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img_1)
    plt.subplot(1, 2, 2)
    plt.imshow(img_2)
    plt.show()

@dataclass
class GaussianTrainerParams:
    name: str = "dummy_params"
    percent_dense = 0.01
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30000
    opacity_lr = 0.05
    scaling_lr = 0.005
    feature_lr = 0.0025
    rotation_lr = 0.001
    saving_iterations = [7000, 30000]
    max_iterations = 30000

@pytest.mark.current 
def test_backward_inno_reprod_size():
    env_config = get_env_config()
    source_gs = GaussianModel(3)
    r = 1.0
    N = 1000
    red = [1, 0, 0]
    blue = [0, 0, 1]
    pcd = sphere_point_cloud(r, N, blue)
    cam = Camera("FlipY")
    source_gs.create_from_pcd(pcd, r)

    cam.lookat(2 * np.array([1, 0, 1]), np.array([0, 0, 0]))

    vanilla_renderer = create_vanilla_renderer(env_config)
    inno_reprod_renderer = create_inno_reprod_renderer(env_config)
    target_img = vanilla_renderer.render(cam, source_gs)["render"]
    target_img = target_img.detach()
    target_img_np = target_img.cpu().numpy().transpose(1, 2, 0)
    target_img_np = target_img_np[::-1, :, :]
    target_img_np = target_img_np.clip(0, 1)

    plt.imshow(target_img_np)
    plt.show()


    gs = GaussianModel(3)
    gs.create_from_pcd(pcd, r, 10.0)

    params = GaussianTrainerParams()
    gs.training_setup(params)
    N_TRAIN = 2000
    N_LOG = 200
    for i in range(1, N_TRAIN+1):
        gs.update_learning_rate(i)
        if i % 1000 == 0:
            gs.oneupSHdegree()
        result_img = inno_reprod_renderer.render(cam, gs)["render"]
        # result_img = vanilla_renderer.render(cam, gs)["render"]
        loss = torch.mean((result_img - target_img) ** 2)
        loss.backward()

        with torch.no_grad():
            if i % N_LOG == 0:
                print(f"iter {i}, loss: {loss.item():.4f}")
                result_img_np = result_img.detach().cpu().numpy().transpose(1, 2, 0)
                result_img_np = result_img_np[::-1, :, :]
                result_img_np = result_img_np.clip(0, 1)
                compare_show(target_img_np, result_img_np)

            gs.optimizer.step()
            gs.optimizer.zero_grad()