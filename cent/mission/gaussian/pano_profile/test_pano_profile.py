import pytest 
from .mission import Mission 
@pytest.mark.current 
def test_profile():
    f = "pano_mip360_bg_first"
    mission = Mission(f + ".json")
    mission.exec()