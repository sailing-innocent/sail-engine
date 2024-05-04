from module.data.point_cloud import sphere_point_cloud
from module.utils.camera.basic import Camera
from module.model.gaussian.vanilla import GaussianModel
import numpy as np 
import torch 
import pytest 
from app.diff_renderer.gaussian_rasterizer.pano import create_gaussian_renderer as create_reprod_renderer
from app.diff_renderer.gaussian_rasterizer.vanilla import create_gaussian_renderer as create_vanilla_renderer 
from mission.config.env import get_env_config
import matplotlib.pyplot as plt
from dataclasses import dataclass

def compare_show(imgs):
    plt.figure()
    N = len(imgs)
    for i, img in enumerate(imgs):
        plt.subplot(1, N, i+1)
        plt.imshow(img)
    plt.show()

@dataclass
class GaussianTrainerParams:
    name: str = "dummy_params"
    percent_dense = 0.01
    position_lr_init = 0.0
    position_lr_final = 0.0
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30000
    opacity_lr = 0.0
    scaling_lr = 0.0
    feature_lr = 0.25
    rotation_lr = 0.0
    saving_iterations = [7000, 30000]
    max_iterations = 30000

@pytest.mark.app 
def test_backward_inno_reprod_color():
    env_config = get_env_config()
    source_gs = GaussianModel(0)
    r = 1.0
    N = 100
    red = [1, 0, 0]
    blue = [0, 0, 1]
    pcd = sphere_point_cloud(r, N, blue)
    cam = Camera("FlipY")

    source_gs.create_from_pcd(pcd, r, 10.0)
    cam.lookat(2 * np.array([1, 0, 1]), np.array([0, 0, 0]))
    cam.set_res(16, 16)
    
    vanilla_renderer = create_vanilla_renderer(env_config)
    reprod_renderer = create_reprod_renderer(env_config)
    target_img = vanilla_renderer.render(cam, source_gs)["render"]
    target_img = target_img.detach()
    target_img_np = target_img.cpu().numpy().transpose(1, 2, 0)
    target_img_np = target_img_np[::-1, :, :]
    target_img_np = target_img_np.clip(0, 1)

    # plt.imshow(target_img_np)
    # plt.show()

    N_TRAIN = 2000
    N_BG_DONE = 1000
    N_LOG = 200
    pcd = sphere_point_cloud(r, N, red)
    gs = GaussianModel(0)
    gs.create_from_pcd(pcd, r, 10.0)
    params = GaussianTrainerParams()
    gs.training_setup(params)

    pano = torch.zeros(3, 16, 32).float().cuda()
    pano.requires_grad = True
    pano_optim = torch.optim.Adam([pano], lr=0.01)

    vanilla_gs = GaussianModel(0)
    vanilla_gs.create_from_pcd(pcd, r, 10.0)
    vanilla_gs.training_setup(params)

    for i in range(1, N_TRAIN+1):
        gs.update_learning_rate(i)
        if i % 1000 == 0:
            gs.oneupSHdegree()
        result_img = reprod_renderer.render(cam, gs, pano)["render"]
        # result_img = vanilla_renderer.render(cam, gs)["render"]
        loss = torch.mean((result_img - target_img) ** 2)
        loss.backward()

        vanilla_result = vanilla_renderer.render(cam, vanilla_gs)["render"]
        vanilla_loss = torch.mean((vanilla_result - target_img) ** 2)
        vanilla_loss.backward()

        with torch.no_grad():
            if i % N_LOG == 0:
                print(f"iter {i}, loss: {loss.item():.4f}")
                result_img_np = result_img.detach().cpu().numpy().transpose(1, 2, 0)
                result_img_np = result_img_np[::-1, :, :]
                result_img_np = result_img_np.clip(0, 1)
                vanilla_result_img_np = vanilla_result.detach().cpu().numpy().transpose(1, 2, 0)[::-1, :, :].clip(0, 1) 
                pano_np = pano.detach().cpu().numpy().transpose(1, 2, 0)[::-1, :, :].clip(0, 1)
                compare_show([target_img_np, result_img_np, vanilla_result_img_np, pano_np])
                # compare_show([target_img_np, result_img_np])
                # print("gausssian features: ", gs.get_features[:10])
            gs.optimizer.step()
            gs.optimizer.zero_grad()

            vanilla_gs.optimizer.step()
            vanilla_gs.optimizer.zero_grad()
            if i < N_BG_DONE:
                pano_optim.step()
                pano_optim.zero_grad()
