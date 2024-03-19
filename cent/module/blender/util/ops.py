import bpy 
import os 

def brender(rootdir: str, filename: str):
    scene = bpy.context.scene 
    scene.render.filepath = os.path.join(rootdir, filename)
    bpy.ops.render.render(write_still=True)
