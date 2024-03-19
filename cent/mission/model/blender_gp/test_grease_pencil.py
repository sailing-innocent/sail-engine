import pytest 

from module.blender.wrapper import blender_executive
from module.blender.util.gpencil import init_gpencil, gp_draw_line

@blender_executive
def draw_grease(rootdir):
    gpencil_layer = init_gpencil()
    gp_frame = gpencil_layer.frames.new(0)
    line = gp_draw_line(gp_frame, (0, 0, 0), (1, 1, 1))

@pytest.mark.current
def test_draw_grease():
    draw_grease(subfolder = "model", filename = "simple_grease_pencil")
    assert True 