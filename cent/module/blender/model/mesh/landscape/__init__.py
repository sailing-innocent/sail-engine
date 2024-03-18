import bpy
from .utils import (
    create_mesh_object,
    gen_grid,
    gen_grid_from_heightmap
)

def create_landscape_from_heightmap(context, heightmap, name="mount"):
    scene = context.scene
    view_layer = context.view_layer
    new_name = name
    verts, faces = gen_grid_from_heightmap(heightmap)
    new_object = create_mesh_object(context, verts, [], faces, new_name)
    new_object.select_set(True)
    return new_object

def create_landscape(context, 
        width = 5,         
        height = 1.0, 
        subdiv = 1024, 
        noise_size = 1.0, 
        lacunarity = 2.0,        
        falloffsize = 1.0,
        name = "Landscape",
        offset_x = 0.0,
        offset_y = 0.0
    ):
    # -- AntAddLandscape(bpy.types.Operator)
    scene = context.scene
    view_layer = context.view_layer
    new_name = name
    # grid mesh 
    subdivision_x = subdiv 
    subdivision_y = subdiv

    mesh_size_x = width * 2 
    mesh_size_y = width * 2

    depth = 8
    dimension = 1.0

    falloff = 3
    falloffsize_x = falloffsize
    falloffsize_y = falloffsize
    edge_level = 0
    props = {
        "noise_size": noise_size,
        "falloff": falloff,
        "meshsize_x": mesh_size_x,
        "meshsize_y": mesh_size_y,
        "falloffsize_x": falloffsize_x,
        "falloffsize_y": falloffsize_y,
        "edge_level": edge_level,
        "lacunarity": lacunarity,
        "depth": depth,
        "dimension": dimension,
        "height": height,
    }
    verts, faces, heightmap = gen_grid(
        subdivision_x, 
        subdivision_y, 
        False, 
        mesh_size_x, 
        mesh_size_y, 
        offset_x,
        offset_y,
        props, 
        False, 
        0.0)
    # create object according to verts
    new_object = create_mesh_object(context, verts, [], faces, new_name)
    print(new_object)
    new_object.select_set(True)
    
    return new_object, heightmap
