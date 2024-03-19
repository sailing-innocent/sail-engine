import pytest 
from module.blender import blender_executive 
import bpy 
import os 
from .scene import Scene

@blender_executive
def extract_scene(rootdir):
    scene = Scene()
    scene.from_blender(bpy.context.scene)

@pytest.mark.current
def test_extract_scene():
    extract_scene(subfolder="scene", filename="dummy", clear=False)
    assert True

