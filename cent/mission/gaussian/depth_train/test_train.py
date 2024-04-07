import pytest 
from .mission import Mission 
@pytest.mark.current 
def test_train():
    # f = "train_depth_mip360d"
    f = "eval_depth_mip360d"
    mission = Mission(f + ".json")
    mission.exec()