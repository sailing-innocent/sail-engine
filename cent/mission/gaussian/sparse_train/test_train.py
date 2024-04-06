import pytest 
from .mission import Mission 
@pytest.mark.current 
def test_train():
    # f = "train_vanilla_basic"
    # f = "train_vanilla_vanilla"
    # f = "train_vanilla_epipolar"
    f = "train_sparse_vanilla"
    mission = Mission(f + ".json")
    mission.exec()