import pytest 
from .mission import Mission 
@pytest.mark.current 
def test_train():
    f = "train_color_vanilla_mip360"
    mission = Mission(f + ".json")
    mission.exec()