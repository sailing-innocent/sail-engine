import pytest 
from .mission import Mission

@pytest.mark.current 
def test_gaussian_reprod_render():
    # f = "render_vanilla"
    # f = "render_inno_reprod"
    f = "render_inno_reprod_sphere"
    mission = Mission(f + ".json")
    mission.exec()