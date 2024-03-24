# Simple Blender Scene

from ..camera.basic import create_basic_camera
from ..light.basic import create_basic_light

from ..model.mesh.primitive import add_uv_sphere, add_cube

import bpy 
import mathutils 

def simple_cube():
    obj = add_cube()
    camera = create_basic_camera(origin=mathutils.Vector((2, 6, 6)), target=obj)
    light  = create_basic_light(origin=mathutils.Vector((0, 5, 5)), target=obj)
    return obj, camera, light