from module.utils.camera.basic import Camera
from module.model.gaussian.vanilla import GaussianModel
import numpy as np 
import torch 
from app.diff_renderer.gaussian_rasterizer.depth import create_gaussian_renderer as create_renderer 
from mission.config.env import get_env_config
from app.trainer.nvs.gs.basic import GaussianTrainerParams
import matplotlib.pyplot as plt

from module.utils.torch.camera_util import DK2Points
from module.utils.pointcloud.io import storePly
from lib.ext.unidepth.utils import colorize, image_grid
from module.dataset.nvs.mip360.dataset import Mip360DatasetConfig, Mip360Dataset
import pytest

@pytest.mark.current 
def test_depth_nerf_blender():
    env_config = get_env_config()
    source_gs = GaussianModel(3)
    # source_gs.load_ply("D:/pretrained/gaussian/nerf_blender_lego_30000.ply")
    source_gs.load_ply("D:/pretrained/gaussian/mip360_bicycle_30000.ply")
    config = Mip360DatasetConfig(env_config)
    dataset = Mip360Dataset(config)
    pair = dataset[0]
    cam = Camera("FlipY")
    cam.from_info(pair.cam)
    w = pair.cam.ResW
    h = pair.cam.ResH
    params = GaussianTrainerParams()
    source_gs.training_setup(params)
    renderer = create_renderer(env_config)
    target_img = torch.ones(4, h, w).float().cuda()

    result_img = renderer.render(cam, source_gs)["render"].detach()
    assert result_img.shape == target_img.shape

    # result_img = result_img.detach().flip(1) # CHW
    
    rgb = result_img[0:3]
    depth = result_img[3]
    # flip
    rgb_img = rgb.cpu().permute(1, 2, 0).numpy().clip(0, 1)
    plt.imshow(rgb_img)
    plt.show()

    intr = torch.from_numpy(cam.info.K).float().cuda()
    points = DK2Points(depth, intr)
    points_np = points.reshape(-1, 3).detach().cpu().numpy()
    color = rgb_img.reshape(-1, 3)

    ply_path = "D:/temp/depth.ply"
    storePly(ply_path, points_np, color)
    # loss = torch.nn.functional.mse_loss(result_img, target_img)
    # loss.backward()
    model = torch.hub.load(
        "lpiccinelli-eth/unidepth",
        "UniDepthV1_ViTL14",
        pretrained=True,
        # trust_repo=True,
        # force_reload=True,
    )
    model = model.to("cuda")
    predictions = model.infer(rgb, intr)

    depth_pred = predictions["depth"].squeeze()
    points_pred = DK2Points(depth_pred, intr)
    points_pred_np = points_pred.reshape(-1, 3).detach().cpu().numpy()

    ply_path = "D:/temp/depth_pred.ply"
    storePly(ply_path, points_pred_np, color)

    # compare depth
    depth_loss = torch.nn.functional.mse_loss(depth, depth_pred)
    print(depth_loss.item())