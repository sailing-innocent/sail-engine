import bpy 
from bpy_extras import object_utils
from ....util.noise import si_noise
import numpy as np 

def create_mesh_object(context, verts, edges, faces, name):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return object_utils.object_data_add(context, mesh, operator=None)

def gen_grid_from_heightmap(heightmap):
    sub_d_x = heightmap.shape[1]
    sub_d_y = heightmap.shape[0]
    verts = []
    faces = []
    vappend = verts.append
    fappend = faces.append 
    meshsize_x = 5
    meshsize_y = 5 
    tri = False

    for i in range(0, sub_d_x):
        x = meshsize_x * ( i / (sub_d_x - 1) - 1 / 2)
        for j in range(0, sub_d_y):
            y = meshsize_y * (j / (sub_d_y - 1) - 1 / 2)
            z = heightmap[i][j]
            vappend((x, y, z))
        
            if i > 0 and j > 0:
                A = i * sub_d_y + (j - 1)
                B = i * sub_d_y + j 
                C = (i - 1) * sub_d_y + j 
                D = (i - 1) * sub_d_y + (j - 1)
                if not tri:
                    fappend((A, B, C, D))
                else:
                    fappend((A, B, D))
                    fappend((B, C, D))

    return verts, faces

def gen_grid(sub_d_x, sub_d_y, tri, meshsize_x, meshsize_y, offset_x, offset_y, props, water_plane, water_level):
    verts = []
    faces = []
    vappend = verts.append
    fappend = faces.append 

    heightmap = np.ones((sub_d_x, sub_d_y), dtype=float)

    for i in range(0, sub_d_x):
        x = meshsize_x * ( i / (sub_d_x - 1) - 1 / 2)
        for j in range(0, sub_d_y):
            y = meshsize_y * (j / (sub_d_y - 1) - 1 / 2)
            if not water_plane:
                z = si_noise((x - offset_x, y - offset_y, 0), props)
            else:
                z = water_level

            # save i, j, z to height field png
            heightmap[i][j] = z / props["height"] * 255.0

            vappend((x, y, z))

            if i > 0 and j > 0:
                A = i * sub_d_y + (j - 1)
                B = i * sub_d_y + j 
                C = (i - 1) * sub_d_y + j 
                D = (i - 1) * sub_d_y + (j - 1)
                if not tri:
                    fappend((A, B, C, D))
                else:
                    fappend((A, B, D))
                    fappend((B, C, D))

    return verts, faces, heightmap