import pytest 
from module.blender.wrapper import blender_executive 
from module.blender.scene.simple import simple_cube
from app.blender_render_engine.dummy import DummyRenderEngine
import bpy 
import os 

@blender_executive
def dummy_render(rootdir):
    # simple_cube()
    bpy.utils.register_class(DummyRenderEngine)
    scene = bpy.context.scene 
    # scene.render.engine = 'CYCLES'
    scene.render.engine = 'DUMMY'
    scene.render.filepath = os.path.join(rootdir, 'dummy_render.png')
    bpy.ops.render.render(write_still=True)

@pytest.mark.current
def test_dummy_render():
    dummy_render(subfolder="render", filename="dummy_render", clear=False)
    assert True