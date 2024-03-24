import bpy 

def add_uv_sphere():
    # create object
    bpy.ops.mesh.primitive_uv_sphere_add(
        location=(0,0,0),
        segments=32,
        ring_count=16,
        radius=1)
    obj = bpy.context.object
    return obj

def add_cube():
    # create object
    bpy.ops.mesh.primitive_cube_add(location=(0,0,0))
    obj = bpy.context.object
    return obj