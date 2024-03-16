import pytest 

from module.data.camera_info import CameraInfo
import numpy as np 

@pytest.mark.current 
def test_camera_info():
    info_a = CameraInfo()
    assert np.all(info_a.R == np.eye(3))
    assert np.all(info_a.T == np.zeros(3))
    assert np.equal(info_a.FovY, 60 / 180 * np.pi)
    assert np.equal(info_a.FovX, 60 / 180 * np.pi)
    assert info_a.ResW == 400
    assert info_a.ResW == 400