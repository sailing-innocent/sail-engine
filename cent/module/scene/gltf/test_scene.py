import pytest 
from .scene import Scene
import json 
import os 

@pytest.mark.current
def test_scene():

    gltf_dir = "E:/asset/scene/dummy"
    gltf_file = os.path.join(gltf_dir, "scene.gltf")

    with open(gltf_file, "r") as f:
        gltf_json = json.load(f)

    scene = Scene()
    scene.from_gltf(gltf_json)
    # gltf_json = scene.to_gltf()
    # print(gltf_json)
    target_json = scene.to_gltf()
    target_file = os.path.join(gltf_dir, "scene_target.gltf")
    with open(target_file, "w") as f:
        json.dump(target_json, f, indent=4)
    
    assert True 