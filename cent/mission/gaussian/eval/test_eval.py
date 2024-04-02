import pytest 
from .mission import Mission

@pytest.mark.current 
def test_gaussian_reprod_render():
    # f = "render_vanilla"
    # f = "eval_vanilla"
    # f = "render_inno_reprod"
    # f = "render_inno_reprod_sphere"
    # f = "render_inno_split"
    # f = "render_inno_torch"
    f = "demo_inno_torch"
    # f = "demo_vanilla"
    # f = "demo_inno_split"
    # f = "demo_inno_reprod"
    mission = Mission(f + ".json")
    mission.exec()