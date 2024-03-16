import pytest 
import numpy as np 
from module.utils.camera.basic import Camera

from module.blender import blender_executive
from module.blender.obj.mesh.arrow import add_arrow

@blender_executive
def vis_rays(rootdir, flipz = True):
    pos = np.array([0, -1, 0])
    target = np.array([0, 0, 0])
    camera = Camera()
    if not flipz:
        camera.flip()
    camera.lookat(pos, target)
    camera.set_res(20, 10)
    orig, dirs = camera.rays 
    assert orig.shape == (3,) # xyz
    assert dirs.shape == (10, 20, 3) # H, W, xyz

    for i in range(camera.info.ResW):
        for j in range(camera.info.ResH):
            u = i / camera.info.ResW
            v = j / camera.info.ResH
            add_arrow(orig, dirs[j, i], name=f"ray_{i}_{j}", color=(u, v, 0, 1))

@pytest.mark.vis
def test_vis_rays():
    vis_rays(subfolder = "debug", filename = "camera_rays_flipz", flipz=True)
    vis_rays(subfolder = "debug", filename = "camera_rays_flipy", flipz=False)
