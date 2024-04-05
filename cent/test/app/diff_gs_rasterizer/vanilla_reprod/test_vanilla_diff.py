from module.data.point_cloud import sphere_point_cloud
from module.utils.camera.basic import Camera
from module.model.gaussian.vanilla import GaussianModel
import numpy as np 
import torch 
from app.diff_renderer.gaussian_rasterizer.vanilla_reprod import create_gaussian_renderer as create_vanilla_renderer 
from mission.config.env import get_env_config
from app.trainer.nvs.gs.basic import GaussianTrainerParams
import matplotlib.pyplot as plt

def test_orth():
    env_config = get_env_config()
    source_gs = GaussianModel(3)
    source_gs.load_ply("D:/pretrained/gaussian/nerf_blender_lego_30000.ply")
    w = 256
    h = 256
    cam = Camera("FlipY")
    cam.lookat(2 * np.array([1, 0, 1]), np.array([0, 0, 0]))
    cam.set_res(w, h)
    params = GaussianTrainerParams()
    source_gs.training_setup(params)
    vanilla_renderer = create_vanilla_renderer(env_config)
    target_img = torch.ones(3, h, w).cuda()
    result_img = vanilla_renderer.render(cam, source_gs)["render"]
    result_img_np = result_img.cpu().detach().numpy().transpose(1, 2, 0)[::-1, :, :].clip(0, 1)
    plt.imshow(result_img_np)
    plt.show()

    loss = torch.nn.functional.mse_loss(result_img, target_img)
    loss.backward()

