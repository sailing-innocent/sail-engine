import bpy 
import mathutils 

from ..util.constraints import track_to_constraints

def create_basic_camera(origin=mathutils.Vector((0.0, 0.0, 2.0)), lens=25, clip_start=0.1, clip_end=100, camera_type='PERSP', ortho_scale=6, target=None):
    # Create object and camera
    camera = bpy.data.cameras.new("Camera")
    camera.lens = lens 
    camera.clip_start = clip_start 
    camera.clip_end = clip_end 
    camera.type = camera_type # 'PERSP', 'ORTHO', 'PANO'
    if (camera_type == 'ORTHO'):
        camera.ortho_scale = ortho_scale

    # Link Object to Scene 
    obj = bpy.data.objects.new("CameraObj", camera)
    obj.location = origin 
    bpy.context.collection.objects.link(obj)
    bpy.context.scene.camera = obj # Make Current
    
    if target: 
        track_to_constraints(obj, target)

    return obj 


