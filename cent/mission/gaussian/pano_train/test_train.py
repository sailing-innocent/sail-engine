import pytest 
from .mission import Mission 
@pytest.mark.current 
def test_train():
    # f = "train_pano_basic"
    # f = "train_pano_vanilla"
    # f = "train_pano_bg_first"
    f = "train_pano_bg_first_mip360"
    mission = Mission(f + ".json")
    mission.exec()