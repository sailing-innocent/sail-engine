import pytest 
from .mission import Mission 
@pytest.mark.current 
def test_train():
    f = "train_pano_basic"
    mission = Mission(f + ".json")
    mission.exec()