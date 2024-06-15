import pytest 
from module.blender.wrapper import blender_executive 
from module.blender.scene.simple import simple_cube
from module.blender.util.ops import brender 
from app.blender_render_engine.dummy import DummyRenderEngine
import bpy 

@blender_executive
def dummy_render(rootdir):
    simple_cube()
    bpy.utils.register_class(DummyRenderEngine)
    scene = bpy.context.scene 
    # scene.render.engine = 'CYCLES'
    scene.render.engine = 'DUMMY'
    brender(rootdir, 'dummy_render.png')

@pytest.mark.current
def test_dummy_render():
    dummy_render(subfolder="render", filename="dummy_render")
    assert True