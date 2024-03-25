import pytest 
from module.blender import blender_executive 
import bpy 
import os 
from module.scene.blender.scene import Scene as BlenderScene
from module.scene.gltf.scene import Scene as GLTFScene
from .scene import Scene
import json 

@blender_executive
def extract_scene(rootdir):
    bscene = BlenderScene()
    bscene.from_blender(bpy.context.scene)
    scene = Scene()
    scene.from_blender(bscene)
    gscene = scene.to_gltf()
    gltf_dir = "E:/asset/scene/dummy"
    gltf_json = gscene.to_gltf(gltf_dir)
    gltf_file = os.path.join(gltf_dir, "scene_blender.gltf")
    with open(gltf_file, "w") as f:
        json.dump(gltf_json, f, indent=4)

@pytest.mark.current
def test_extract_scene():
    extract_scene(subfolder="scene", filename="dummy", clear=False)
    assert True

