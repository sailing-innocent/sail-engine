import pytest 
from .mission import Mission 
@pytest.mark.current 
def test_profile():
    # f = "depth_mip360d_train"
    # f = "depth_mip360d_train_temp"
    # f = "depth_mip360d_basic_train"
    f = "depth_mip360d_zzh_diff_param_train"
    mission = Mission(f + ".json")
    mission.exec()