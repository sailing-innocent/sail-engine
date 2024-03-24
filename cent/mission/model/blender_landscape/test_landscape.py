import pytest 
from module.blender.wrapper import blender_executive
from module.blender.scene.landscape import create_landscape_scene_from_heightmap, create_voronoi_scene_landscape, create_simple_mountain
from module.blender.util.ops import brender 

@blender_executive
def blender_simple_mountain(rootdir):
    width = 5
    subdiv = 1024
    noise_size = 1.0
    lacunarity = 2.0
    height = 1.0
    falloffsize = 0.8
    name = "mf_2"
    offset_x = 0.0
    offset_y = 0.0
    mount, camera, light, heightmap = create_simple_mountain(width, height, subdiv, noise_size, lacunarity, falloffsize, name, offset_x, offset_y)
    brender(rootdir, 'simple_mountain.png')

@pytest.mark.current 
def test_landscape():
    blender_simple_mountain(subfolder="landscape", filename="simple_mountain")
    assert True