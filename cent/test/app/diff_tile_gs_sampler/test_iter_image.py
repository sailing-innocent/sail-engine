import pytest 

from module.model.gaussian.model import Gaussians2D, Gaussians2DTrainArgs
from app.diff_renderer.gaussian_sampler.inno import GaussianSampler
import torch 
import numpy 
import cv2 as cv
import matplotlib.pyplot as plt

@pytest.mark.current 
def test_sampler_2d():
    sampler = GaussianSampler()
    save_dir = "D:/workspace/data/result/tile_gs_sampler"
    img_path = '../doc/latex/figure/asset/asset_zzh_logo.png'
    # read image
    target_img_np = cv.imread(img_path, cv.IMREAD_COLOR)
    # BGR -> RGB
    target_img_np = target_img_np[:, :, ::-1]
    # print(target_img_np)
    width = target_img_np.shape[1]
    height = target_img_np.shape[0]
    plt.imshow(target_img_np)
    plt.show()
    plt.imsave(f'{save_dir}/target.png', target_img_np)
    target_img = torch.tensor(target_img_np.transpose(2, 0, 1) / 255.0, dtype=torch.float32).cuda()
    # flip y
    train_args = Gaussians2DTrainArgs()
    gaussians = Gaussians2D.random(100, 5)
    gaussians.requires_grad()
    gaussians.training_setup(train_args)

    N_ITER = 200
    N_LOG = 30
    N_DELAY = 50
    for i in range(N_ITER):
        result_img = sampler.sample(gaussians, width, height)
        loss = torch.functional.F.mse_loss(target_img, result_img)
        loss.backward()

        with torch.no_grad():
            gaussians.optim.step()
            gaussians.optim.zero_grad()
            if i % N_LOG == 0:
                print(f'Iter {i}, loss: {loss.item()}')
                result_img_np = result_img.detach().cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)
                # compare
                plt.subplot(1, 2, 1)
                plt.imshow(target_img_np)
                plt.subplot(1, 2, 2)
                plt.imshow(result_img_np)
                plt.show()
                plt.imsave(f'{save_dir}/result_{i}.png', result_img_np)
