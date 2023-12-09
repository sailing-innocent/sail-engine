import pytest 
from .mission import Mission 
@pytest.mark.current 
def test_train():
    # f = "train_vanilla_basic"
    f = "train_vanilla_vanilla"
    # f = "train_vanilla_nerf_blender_full"
    # f = "train_vanilla_mip360"
    # f = "train_vanilla_mip360_full"
    # f = "train_vanilla_tank_temple"
    # f = "train_vanilla_tank_temple_full"

    # f = "train_reprod_mip360_full"
    # f = "train_reprod_noxyz_sample"

    # f = "train_inno_reprod_sample"
    # f = "train_vanilla_plain"

    # f = "eval_vanilla_sample"
    # f = "eval_vanilla_mip360"
    # f = "eval_vanilla_mip360_full"
    mission = Mission(f + ".json")
    mission.exec()