# from https://github.com/5agado/data-science-learning

import bpy 
from bpy.types import GPencilFrame

def get_grease_pencil(
    gpencil_obj_name: str = 'GPencil',
    clear_data: bool = False) -> bpy.types.GreasePencil:  
    """
    Return Grease Pencil Object with Given Name
    init one of not exist
    """

    # delete if exist
    if (gpencil_obj_name in bpy.context.scene.objects and clear_data):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.scene.objects[gpencil_obj_name].select_set(True)
        bpy.ops.object.delete(use_global=True)

    # init if not exist
    if (gpencil_obj_name not in bpy.context.scene.objects):
        bpy.ops.object.gpencil_add(
            location=(0, 0, 0),
            type='EMPTY'
        )
        bpy.context.object.name = gpencil_obj_name
    
    return bpy.context.scene.objects[gpencil_obj_name]

def get_gpencil_layer(
    gpencil: bpy.types.GreasePencil,
    gpencil_layer_name: str = 'GP_Layer',
    clear_layer = False):
    """
    Return Grease Pencil Layer with Given Name
    init one of not exist
    """
    if gpencil.data.layers and gpencil_layer_name in gpencil.data.layers:
        gpencil_layer = gpencil.data.layers[gpencil_layer_name]
    else:
        # new
        gpencil_layer = gpencil.data.layers.new(gpencil_layer_name, set_active=True)

    if clear_layer:
        gpencil_layer.clear()

    return gpencil_layer 


def init_gpencil(
    gpencil_obj_name: str = 'GPencil',
    gpencil_layer_name: str = 'GP_Layer',
    clear_layer: bool = True
) -> bpy.types.GPencilLayer:
    gpencil = get_grease_pencil(gpencil_obj_name)
    gpencil_layer = get_gpencil_layer(gpencil, gpencil_layer_name, clear_layer)
    return gpencil_layer

def gp_draw_line(gp_frame: GPencilFrame, p0: tuple, p1: tuple, material_index: int = 0):
    gp_stroke = gp_frame.strokes.new()
    gp_stroke.display_mode = '3DSPACE'
    gp_stroke.material_index = material_index 

    # Define Stroke Geometry
    gp_stroke.points.add(count = 2)
    gp_stroke.points[0].co = p0
    gp_stroke.points[1].co = p1
    return gp_stroke