from .mission import Mission

def test_compare():
    f = "compare_diff_render_full"
    m = Mission(f + ".json")
    m.exec()