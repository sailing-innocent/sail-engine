from module.utils.camera.basic import Camera
from module.model.gaussian.vanilla import GaussianModel
import numpy as np 
import torch 
from app.diff_renderer.gaussian_rasterizer.depth import create_gaussian_renderer as create_renderer 
from mission.config.env import get_env_config
from app.trainer.nvs.gs.basic import GaussianTrainerParams
import matplotlib.pyplot as plt

def test_depth_diff():
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
    renderer = create_renderer(env_config)
    target_img = torch.ones(4, h, w).float().cuda()

    result_img = renderer.render(cam, source_gs)["render"]
    assert result_img.shape == target_img.shape
    result_img_np = result_img.cpu().detach().numpy().transpose(1, 2, 0)[::-1, :, 0:3].clip(0, 1)
    plt.imshow(result_img_np)
    plt.show()
    # loss = torch.nn.functional.mse_loss(result_img, target_img)
    # loss.backward()

