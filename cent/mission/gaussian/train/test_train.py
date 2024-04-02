import pytest 
from .mission import Mission 
@pytest.mark.current 
def test_train():
    # f = "train_vanilla_basic"
    # f = "train_vanilla_vanilla"
    # f = "train_vanilla_only_trick"
    # f = "train_inno_reprod_basic"
    # f = "train_inno_reprod_vanilla"
    # f = "train_inno_torch_basic"
    f = "train_inno_torch_vanilla"
    mission = Mission(f + ".json")
    mission.exec()