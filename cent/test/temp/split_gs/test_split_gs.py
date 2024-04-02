import pytest 

from module.model.gaussian.model import Gaussians2D, Gaussians2DTrainArgs
from app.diff_renderer.gaussian_sampler.inno import GaussianSampler
from app.diff_renderer.gaussian_projector.inno import GaussianProjector

from module.dataset.nvs.blender.dataset import create_dataset as create_nerf_blender_dataset
from mission.config.env import get_env_config
from module.utils.camera.basic import Camera
from module.model.gaussian.zzh import GaussianModel
from module.data.point_cloud import sphere_point_cloud

import torch 
import matplotlib.pyplot as plt

def show_compare_img(img1, img2, path=None):
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    if path is not None:
        plt.savefig(path)
    # plt.show()

@pytest.mark.current
def test_split_gs():
    sampler = GaussianSampler()
    projector = GaussianProjector()
    env_config = get_env_config()
    dataset = create_nerf_blender_dataset(env_config, 'lego', 'train')
    N_sparse = 10
    pairs = dataset.pairs(N_sparse, True)
    gs = GaussianModel(3)
    r = 1.0
    N = 10000
    N_batch = 5
    red = [1, 0, 0]
    blue = [0, 0, 1]
    pcd = sphere_point_cloud(r, N, blue)
    gs.create_from_pcd(pcd, r)
    gs.save_ply('D:/temp/before.ply')

    N_ITER = 10
    N_SHOW = 2
    N_SUB_ITER = 200
    N_SUB_LOG = 350
    cam_infos = []
    cams = []
    imgs = []
    target_imgs = []    
    gs2ds = []

    train_args = Gaussians2DTrainArgs()

    for idx in range(N_ITER):
        # indicies = torch.randint(0, N_sparse, (N_batch,))
        for i in range(N_batch):
            # cam_infos.append(pairs[indicies[i]].cam)
            # imgs.append(pairs[indicies[i]].img)
            cam_infos.append(pairs[i].cam)
            imgs.append(pairs[i].img)
            cam = Camera("FlipY")
            cam.from_info(cam_infos[i])
            cam.set_res(imgs[i].W, imgs[i].H)
            cams.append(cam)
            target_img = torch.tensor(imgs[i].data.transpose(2, 0, 1), dtype=torch.float32).cuda()
            target_img._requires_grad = False
            target_imgs.append(target_img)

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
                    show_compare_img(imgs[i].data, result_img_np, f"D:/temp/{idx}_{i}.png")

            print("epipolar view")
            gs.epipolar_view(gs2ds, cams)
    gs.save_ply("D:/temp/after.ply")