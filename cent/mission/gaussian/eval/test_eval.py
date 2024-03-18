import pytest 
from .mission import Mission

@pytest.mark.current 
def test_gaussian_reprod_render():
    # cfg = "render_vanilla"
    # cfg = "render_inno_reprod"
    # cfg = "render_inno_zzh"
    # cfg = "render_inno_split"
    # json_file_path = "render_inno_light.json"

    # json_file_path = "render_test.json"
    # json_file_path = "render_ing.json"

    # cfg = "demo_reprod"
    # cfg = "demo_reprod_noxyz"
    # cfg = "demo_vanilla"

    # json_file_path = "demo_inno.json"
    # json_file_path = "demo_test.json"
    # json_file_path = "demo_ing.json"

    # cfg = "eval_reprod"
    cfg = "eval_vanilla"
    # json_file_path = "eval_local.json"

    mission = Mission(cfg + ".json")
    mission.exec()