import pytest 

from module.model.gaussian.model import Gaussians2D
from app.diff_renderer.gaussian_sampler.inno import GaussianSampler
from app.diff_renderer.gaussian_projector.inno import GaussianProjector

from module.dataset.nvs.blender.dataset import create_dataset as create_nerf_blender_dataset
from mission.config.env import get_env_config
from module.utils.camera.basic import Camera
from module.model.gaussian.vanilla import GaussianModel
from module.data.point_cloud import sphere_point_cloud

import torch 
import numpy 
import cv2 as cv
import matplotlib.pyplot as plt

def show_compare_img(img1, img2):
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.show()

@pytest.mark.current
def test_sampler_2d():
    sampler = GaussianSampler()
    projector = GaussianProjector()
    env_config = get_env_config()
    dataset = create_nerf_blender_dataset(env_config, 'lego', 'train')
    pairs = dataset.pairs(2, False)

    cam_01 = pairs[0].cam
    img_01 = pairs[0].img
    target_img_01 = torch.tensor(img_01.data.transpose(2, 0, 1), dtype=torch.float32).cuda()
    target_img_01._requires_grad = False    

    cam_02 = pairs[1].cam
    img_02 = pairs[1].img

    target_img_02 = torch.tensor(img_02.data.transpose(2, 0, 1), dtype=torch.float32).cuda()
    target_img_02._requires_grad = False
    
    gs = GaussianModel(3)
    r = 1.0
    N = 1000
    red = [1, 0, 0]
    blue = [0, 0, 1]
    pcd = sphere_point_cloud(r, N, blue)
    gs.create_from_pcd(pcd, r)

    cam01 = Camera("FlipY")
    cam01.from_info(cam_01)
    cam01.set_res(img_01.W, img_01.H)
    gs2d_01 = projector.project(gs, cam01)
    gs2d_01.clone_detach()
    gs2d_01.requires_grad()
    result_img_01 = sampler.sample(gs2d_01, img_01.W, img_01.H, cam01.info.FovY)
    result_img_01_np = result_img_01.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    show_compare_img(img_01.data, result_img_01_np)

    # backward gs2d_01
    N_ITER = 200
    N_LOG = 50
    optim_01 = torch.optim.AdamW(gs2d_01.parameters(), lr=1e-2)

    for i in range(N_ITER):
        result_img_01 = sampler.sample(gs2d_01, img_01.W, img_01.H, cam01.info.FovY)
        loss = torch.functional.F.mse_loss(target_img_01, result_img_01)
        loss.backward(retain_graph=True)
        with torch.no_grad():
            optim_01.step()
            optim_01.zero_grad()
            if i % N_LOG == 0:
                print(f'Iter {i}, loss: {loss.item()}')
                result_img_np = result_img_01.detach().cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)
                # compare
                plt.subplot(1, 2, 1)
                plt.imshow(img_01.data)
                plt.subplot(1, 2, 2)
                plt.imshow(result_img_np)
                plt.show()

    cam02 = Camera("FlipY")
    cam02.from_info(cam_02)
    cam02.set_res(img_02.W, img_02.H)
    gs2d_02 = projector.project(gs, cam02)
    gs2d_02.clone_detach()
    gs2d_02.requires_grad()
    result_img_02 = sampler.sample(gs2d_02, img_02.W, img_02.H, cam02.info.FovY)
    result_img_02_np = result_img_02.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    show_compare_img(img_02.data, result_img_02_np)

    optim_02 = torch.optim.AdamW(gs2d_02.parameters(), lr=1e-2)

    for i in range(N_ITER):
        result_img_02 = sampler.sample(gs2d_02, img_02.W, img_02.H, cam02.info.FovY)
        loss = torch.functional.F.mse_loss(target_img_02, result_img_02)
        loss.backward(retain_graph=True)
        with torch.no_grad():
            optim_02.step()
            optim_02.zero_grad()
            if i % N_LOG == 0:
                print(f'Iter {i}, loss: {loss.item()}')
                result_img_np = result_img_02.detach().cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)
                # compare
                plt.subplot(1, 2, 1)
                plt.imshow(img_02.data)
                plt.subplot(1, 2, 2)
                plt.imshow(result_img_np)
                plt.show()

    # validate result
    result_img_01 = sampler.sample(gs2d_01, img_01.W, img_01.H, cam01.info.FovY)
    result_img_01_np = result_img_01.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    show_compare_img(img_01.data, result_img_01_np)
    result_img_02 = sampler.sample(gs2d_02, img_02.W, img_02.H, cam02.info.FovY)
    result_img_02_np = result_img_02.detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
    show_compare_img(img_02.data, result_img_02_np)

    V1 = cam01.view_matrix()
    V2 = cam02.view_matrix()
    # V1[0][0] x0 + V1[0][1] y0 + V1[0][2] z0 = x1
    # V1[1][0] x0 + V1[1][1] y0 + V1[1][2] z0 = y1

    # V2[0][0] x0 + V2[0][1] y0 + V2[0][2] z0 = x2
    # V2[1][0] x0 + V2[1][1] y0 + V2[1][2] z0 = y2

    # estimate new x0, y0, z0
    # (X^TX)^-1 X^T Y

    Y1 = gs2d_01.means_2d.reshape(N, 2, 1)
    Y2 = gs2d_02.means_2d.reshape(N, 2, 1)
    Y = torch.cat([Y1, Y2], dim=2)
    V1_01 = torch.stack(V1[:, 0], V1[:, 1])
    V1_2 = V1[:, 2]
    C1 = V1_2 * Y - V1_01
    V2_02 = torch.stack(V2[:, 0], V2[:, 1])
    V2_2 = V2[:, 2]
    C2 = V2_2 * Y - V2_02
    X = torch.inverse(C1.transpose(1, 2) @ C1) @ C1.transpose(1, 2) @ C2


    