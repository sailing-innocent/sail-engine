import pytest 
import numpy as np 
from module.utils.camera.basic import Camera

from module.blender import blender_executive
from module.blender.util.camera import create_basic_camera
# bpy mathutils
from mathutils import Matrix 

@blender_executive
def vis_camera(rootdir):
    bcam = create_basic_camera()
    pos = np.array([-1, -1, 0])
    target = np.array([0, 0, 0])
    camera = Camera()
    camera.lookat(pos, target)
    c2w = Matrix(camera.view_to_world())
    bcam.matrix_world = c2w

@pytest.mark.vis 
def test_vis_camera_flipz():
    vis_camera(subfolder = "debug", filename = "create_camera")
