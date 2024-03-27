from module.data.point_cloud import sphere_point_cloud
from module.model.gaussian.vanilla import GaussianModel
from module.utils.camera.basic import Camera
from mission.config.env import get_env_config
from app.diff_renderer.gaussian_projector.inno import GaussianProjector

import numpy as np 
import pytest 

@pytest.mark.app
def test_epipolar():
    env_config = get_env_config()
    source_gs = GaussianModel(3)
    projector = GaussianProjector()
    r = 1.0
    N = 5
    red = [1, 0, 0]
    blue = [0, 0, 1]
    pcd = sphere_point_cloud(r, N, blue)
    source_gs.create_from_pcd(pcd, r)
    cam01 = Camera("FlipY")
    cam01.lookat(2 * np.array([0, -1, 0]), np.array([0, 0, 0]))
    gs01 = projector.project(source_gs, cam01)
    cam02 = Camera("FlipY")
    cam02.lookat(2 * np.array([1, 1, 1]), np.array([0, 0, 0]))
    gs02 = projector.project(source_gs, cam01)
    # print(source_gs.get_xyz)
    # print(source_gs.get_features)
    # print(source_gs.get_scaling)
    # print(source_gs.get_rotation)
    # print(source_gs.get_opacity)

    # print(gs01.means_2d)
    # print(gs01.covs_2d)
    # print(gs01.depth_features)
    # print(gs01.color_features)
    # print(gs02.means_2d)
    # print(gs02.covs_2d)
    # print(gs02.depth_features)
    # print(gs02.color_features)

