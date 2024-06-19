# Visualize the Image As a Plane attached with material

import bpy, bmesh 
from mathutils import Matrix
from bpy_extras.image_utils import load_image

from ..camera.basic import create_basic_camera
import numpy as np 

def vis_view(img_src: np.array, c2w: np.array = np.eye(4), name="img"):
    height, width, n_channels = img_src.shape
    if n_channels == 3:
        img_src = np.concatenate([img_src, np.ones((height, width, 1))], axis=2)
    
    # HWC->WHC
    img_src = img_src.flatten().astype(np.float32)
    assert img_src.shape[0] == width * height * 4
    # show image in blender
    img = bpy.data.images.new(name, width=width, height=height)
    img.pixels = img_src
    # flip image
    img.pack()

    # init the plane on x-y plane in camera coordinate
    bpy.ops.mesh.primitive_plane_add(
        size=1, enter_editmode=False, location=(0, 0, 0))
    plane = bpy.context.object
    # rename
    plane.name = "img_plane_" + name
    
    # transform from camera-coordinate to world-coordinate
    mesh = plane.data
    c2w = Matrix(c2w)
    mesh.transform(c2w)
    bcam = create_basic_camera()
    bcam.matrix_world = c2w

    mat = bpy.data.materials.new("render_result_mat_" + name)
    mat.use_nodes = True
    plane.data.materials.append(mat)
    mat_output = mat.node_tree.nodes['Material Output']
    img_node = mat.node_tree.nodes.new(type="ShaderNodeTexImage")
    emit_node = mat.node_tree.nodes.new(type='ShaderNodeEmission')
    mat.node_tree.links.new(
        img_node.outputs['Color'], emit_node.inputs['Color'])
    mat.node_tree.links.new(emit_node.outputs[0], mat_output.inputs[0])
    img_node.image = img

    return plane 

