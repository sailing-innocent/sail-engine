import pytest 
from .mission import Mission

@pytest.mark.current 
def test_gaussian_reprod_render():
    f = "render_pano"
    mission = Mission(f + ".json")
    mission.exec()