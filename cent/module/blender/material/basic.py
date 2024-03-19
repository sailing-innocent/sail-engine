import bpy

# Simple BRDF
def create_basic_material(color):
    mat = bpy.data.materials.new('BasicMaterial')
    mat.use_nodes = True 
    node = mat.node_tree.nodes[0]

    # Principled BRDF Node
    # - BaseColor
    # - Subsurface
    # - Subsurface Radius
    # - Metallic
    # - Specular
    # - Specular Tint
    # - Roughness
    # - Anisotropic
    node.inputs[0].default_value = color # base color
    node.inputs[4].default_value = 0.9 # Metalic 
    node.inputs[7].default_value = 0.3 # Roughness

    return mat

def create_transparent_material(color: tuple):
    mat = bpy.data.materials.new('TransparentMaterial')
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    material_output = mat.node_tree.nodes.get('Material Output')
    principled_BSDF = mat.node_tree.nodes.get('Principled BSDF')
    principled_BSDF.inputs[0].default_value = color
    principled_BSDF.inputs[21].default_value = 0.3
    return mat

def create_volume_absorption_material(color: tuple):
    mat = bpy.data.materials.new('VolumeAbsorptionMaterial')
    mat.use_nodes = True 
    # delete default node
    node = mat.node_tree.nodes[0]
    mat.node_tree.nodes.remove(node)
    # create Volume Absorption Node
    node = mat.node_tree.nodes.new('ShaderNodeVolumeAbsorption')
    node.inputs[0].default_value = color
    # create Output Node
    material_output = mat.node_tree.nodes.get("Material Output")
    mat.node_tree.links.new(node.outputs[0], material_output.inputs[1])

    return mat