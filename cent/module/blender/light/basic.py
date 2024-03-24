import bpy
from ..util.constraints import track_to_constraints

def create_basic_light(origin, type='POINT', energy=1000, color=(1,1,1), target=None):
    # Light types: 'POINT', 'SUN', 'SPOT', 'HEMI', 'AREA'
    bpy.ops.object.add(type='LIGHT', location=origin)
    obj = bpy.context.object
    obj.data.type = type
    obj.data.energy = energy
    obj.data.color = color
    if target: 
        track_to_constraints(obj, target)
    return obj
