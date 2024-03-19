# Wrapper for Blender Noise

import bpy 

from mathutils.noise import (
    noise, 
    multi_fractal    
)
from math import (
    floor, sqrt,
    sin, cos, pi,
)

noise_basis_default = "BLENDER"
noise_basis = [
#      key    ,   title  ,     help              , index
    ("BLENDER", "Blender", "Blender default noise", 0),
    ("PERLIN_ORIGINAL", "Perlin", "Perlin noise", 1)
]

# Height Scale

def scale(val, iscale, offset, invert):
    if invert != 0:
        return (1.0 - val) * iscale + offset 
    else:
        return val * iscale + offset  

def si_noise(coord, props):
    x, y, z = coord

    # origin 
    x_offset = 1.0
    y_offset = 1.0
    z_offset = 0.0
    origin = x_offset, y_offset, z_offset
    origin_x = x_offset
    origin_y = y_offset
    origin_z = z_offset
    o_range = 1.0

    nsize = props["noise_size"]
    size_x = 1.0
    size_y = 1.0
    size_z = 1.0
    ncoords = ( 
        x / (nsize * size_x) + origin_x, 
        y / (nsize * size_y) + origin_y,
        z / (nsize * size_z) + origin_z)

    # multi_fractal
    nbasis = noise_basis_default
    value = multi_fractal(
        ncoords, 
        props["dimension"], 
        props["lacunarity"], 
        props["depth"], 
        noise_basis=nbasis) * 0.5
    
    height = props["height"]
    value = scale(value, height, 0.0, False)

    # Edge Falloff
    meshsize_x = props["meshsize_x"]
    meshsize_y = props["meshsize_y"]
    falloff = props["falloff"]
    falloffsize_x = props["falloffsize_x"]
    falloffsize_y = props["falloffsize_y"]
    edge_level = props["edge_level"]

    ratio_x, ratio_y = abs(x) * 2 / meshsize_x , abs(y) * 2 / meshsize_y
    fallofftypes = [
        0, 
        sqrt(ratio_y ** falloffsize_y),
        sqrt(ratio_x ** falloffsize_x),
        sqrt(ratio_x ** falloffsize_x + ratio_y ** falloffsize_y)
    ]

    dist = fallofftypes[falloff]
    value -= edge_level 

    if (dist < 1.0):
        dist = (dist * dist * (3 - 2 * dist))
        value = (value - value * dist) + edge_level
    else:
        value = edge_level

    return value