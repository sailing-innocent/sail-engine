import pytest 

from module.model.gaussian.model import Gaussians2D
from app.diff_renderer.gaussian_sampler.inno import GaussianSampler
import torch 
import numpy 
import cv2 as cv
import matplotlib.pyplot as plt

@pytest.mark.current 
def test_sampler_2d():
    sampler = GaussianSampler()
    img_path = '../doc/latex/figure/asset/asset_zzh_logo.png'
    target_img_np = cv.imread(img_path)
    width = target_img_np.shape[1]
    height = target_img_np.shape[0]
    plt.imshow(target_img_np)
    plt.show()
    target_img = torch.tensor(target_img_np.transpose(2, 0, 1) / 255.0, dtype=torch.float32).cuda()
    # flip y
    gaussians = Gaussians2D.random(1000, 8)
    gaussians.requires_grad()

    N_ITER = 100
    N_LOG = 10

    optim = torch.optim.AdamW(gaussians.parameters(), lr=1e-2)
   
    for i in range(N_ITER):
        result_img = sampler.sample(gaussians, width, height)
        loss = torch.functional.F.mse_loss(target_img, result_img)
        loss.backward()
        optim.step()
        optim.zero_grad()
        if i % N_LOG == 0:
            print(f'Iter {i}, loss: {loss.item()}')
            result_img_np = result_img.detach().cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)
            # compare
            plt.subplot(1, 2, 1)
            plt.imshow(target_img_np)
            plt.subplot(1, 2, 2)
            plt.imshow(result_img_np)
            plt.show()