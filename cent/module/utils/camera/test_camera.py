import pytest 

from module.utils.camera.basic import Camera 

import numpy as np 

@pytest.mark.app
def test_camera_default():
    cam = Camera() # FlipZ
    assert np.all(cam.info.R == np.eye(3))
    assert np.all(cam.info.T == np.zeros(3))
    assert np.equal(cam.info.FovY, 60 / 180 * np.pi)
    assert np.equal(cam.info.FovX, 60 / 180 * np.pi)
    assert cam.info.ResW == 400
    assert cam.info.ResH == 400

    pos = np.array([0, -1, 0])
    target = np.array([0, 0, 0])
    cam.lookat(pos, target)
    view_mat = cam.view_matrix
    v = np.array([0, 0, 1, 1])
    target_view_mat = np.eye(4)
    target_view_mat[1][1] = 0
    target_view_mat[2][2] = 0
    target_view_mat[1][2] = 1
    target_view_mat[2][1]= -1
    target_view_mat[2][3] = -1

    assert np.all(view_mat == target_view_mat)
    tv = np.matmul(view_mat, v)
    target_v = np.array([0, 1, -1, 1])
    assert np.all(tv == target_v)

    cam.flip()

    view_mat = cam.view_matrix

    target_view_mat = np.eye(4)
    target_view_mat[1][1] = 0
    target_view_mat[2][2] = 0
    target_view_mat[1][2] = -1
    target_view_mat[2][1]= 1    
    target_view_mat[2][3] = 1

    tv = np.matmul(view_mat, v)
    target_v = np.array([0, -1, 1, 1])
    assert np.all(tv == target_v)
    assert np.all(view_mat == target_view_mat)
