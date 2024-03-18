import bpy
from mathutils import Vector, Matrix
import numpy as np 

def add_arrow(location = (0, 0, 0), direction = (0, 0, 1), length = 1, radius = 0.2, name="Arrow", color=(0.8, 0.1, 0.8, 1)):
    cylinder_length = 0.8
    cone_length = 0.2
    cylinder_radius = 0.1
    cone_radius = 0.15
    bpy.ops.mesh.primitive_cylinder_add(radius=cylinder_radius, depth=cylinder_length)
    cylinder = bpy.context.object
    bpy.ops.mesh.primitive_cone_add(radius1=cone_radius, radius2=0, depth=cone_length)
    cone = bpy.context.object
    cone.location += Vector([0, 0, cylinder_length + cone_length/2])
    cylinder.location += Vector([0, 0, cylinder_length/2])
    # join
    bpy.ops.object.select_all(action='DESELECT')
    cylinder.select_set(True)
    cone.select_set(True)
    bpy.context.view_layer.objects.active = cylinder
    bpy.ops.object.join()
    # rename
    cylinder.name = name
    arrow = bpy.context.object

    # set origin
    arrow.location = Vector(location)
    mw = arrow.matrix_world
    arrow.data.transform(mw)
    # scale
    arrow.scale = Vector([radius, radius, length])

    # set direction
    pi = 3.1415926
    direction = Vector(direction).normalized()
    # z to direction
    z = Vector([0, 0, 1])
    new_x = z.cross(direction)
    new_y = -new_x.cross(direction)
    new_z = direction
    # print()
    # print("new_x: ", new_x)
    # print("new_y: ", new_y)
    # print("new_z: ", new_z)
    # rotation matrix
    R = Matrix([new_x, new_y, new_z]).transposed()
    # apply
    arrow.rotation_euler = R.to_euler('XYZ')

    # material 
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    arrow.data.materials.append(mat)

    return cylinder
