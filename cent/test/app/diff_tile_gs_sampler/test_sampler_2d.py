import pytest 

from module.model.gaussian.model import Gaussians2D
from app.diff_renderer.gaussian_sampler.inno import GaussianSampler
import torch 
import numpy 
import matplotlib.pyplot as plt


@pytest.mark.app
def test_sampler_2d():
    sampler = GaussianSampler()
    width = 512
    height = 512
    # default_gaussians = Gaussians2D(20)
    # target_img = sampler.sample(default_gaussians, width, height)
    # assert target_img.shape == (3, height, width)
    # target_img_np = target_img.detach().cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)[::-1, :, :]
    # plt.imshow(target_img_np)
    # plt.show()

    gaussians = Gaussians2D.random(100, 4)
    # gaussians.requires_grad()
    gs1 = gaussians.subview(0, 25)
    target_img = sampler.sample(gs1, width, height)
    target_img_np = target_img.detach().cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)[::-1, :, :]
    plt.imshow(target_img_np)
    plt.show()

    gs2 = gaussians.subview(75, 100)
    target_img = sampler.sample(gs2, width, height)
    target_img_np = target_img.detach().cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)[::-1, :, :]
    plt.imshow(target_img_np)
    plt.show()
    
    # N_ITER = 100
    # N_LOG = 10

    # optim = torch.optim.AdamW(gaussians.parameters(), lr=1e-2)
   
    # for i in range(N_ITER):
    #     result_img = sampler.sample(gaussians, width, height)
    #     loss = torch.functional.F.mse_loss(target_img, result_img)
    #     loss.backward()
    #     optim.step()
    #     optim.zero_grad()
    #     if i % N_LOG == 0:
    #         print(f'Iter {i}, loss: {loss.item()}')
    #         result_img_np = result_img.detach().cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)[::-1, :, :]
    #         # compare
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(target_img_np)
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(result_img_np)
    #         plt.show()