from ..camera.basic import create_basic_camera
from ..light.basic import create_basic_light 
from ..material.basic import create_basic_material

from ..model.mesh.landscape import create_landscape, create_landscape_from_heightmap
from ..model.mesh.landscape.voronoi import create_voronoi_landscape
import bpy 
import mathutils 

def create_simple_mountain(width, height, subdiv, noise_size, lacunarity, falloffsize, name, offset_x=0.0, offset_y=0.0, context=bpy.context):
    obj, heightmap = create_landscape(context, width, height, subdiv, noise_size, lacunarity, falloffsize, name, offset_x, offset_y)
    mat = create_basic_material((158/256, 122/256, 122/256, 1))
    obj.data.materials.append(mat)
    camera = create_basic_camera(origin=mathutils.Vector((0, 4.5, 1.0)), target=obj)
    light = create_basic_light(origin=mathutils.Vector((3, 0, 3)), target=obj)
    return obj, camera, light, heightmap

def create_landscape_scene_from_heightmap(heightmap):
    obj = create_landscape_from_heightmap(bpy.context, heightmap)
    mat = create_basic_material((158/256, 122/256, 122/256, 1))
    obj.data.materials.append(mat)
    camera = create_basic_camera(origin=mathutils.Vector((0, 2.5, 0)), target=obj)
    light = create_basic_light(origin=mathutils.Vector((0, 5, 5)), target=obj)
    return obj, camera, light

def create_voronoi_scene_landscape():
    obj = create_voronoi_landscape()
    camera = create_basic_camera(origin=mathutils.Vector((0, 2.5, 0)), target=obj)
    light = create_basic_light(origin=mathutils.Vector((0, 5, 5)), target=obj)
    return obj, camera, light