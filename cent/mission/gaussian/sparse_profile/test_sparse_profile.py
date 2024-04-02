import pytest 
from .mission import Mission 
@pytest.mark.current 
def test_profile():
    # f = "vanilla_basic_diff_init_train"
    # f = "vanilla_vanilla_diff_init_train"
    f = "vanilla_diff_trainer_train"
    mission = Mission(f + ".json")
    mission.exec()