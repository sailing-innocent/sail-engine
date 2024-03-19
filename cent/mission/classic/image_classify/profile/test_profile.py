import pytest 
from .mission import Mission

def test_profile_image_classify():
    f = "profile_image_fmnist_mlp_simple"
    m = Mission(f + ".json")
    m.exec()