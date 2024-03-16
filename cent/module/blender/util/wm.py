import bpy 
import os 

def bopen(mainfile_path='sample.blend', clear=True):
    # INIT SAVE
    if (not os.path.exists(mainfile_path)):
        bpy.ops.wm.save_mainfile(filepath=mainfile_path)

    bpy.ops.wm.open_mainfile(filepath = mainfile_path)
    if (clear):
        bclear()

def bclose(mainfile_path='sample.blend'):
    bpy.ops.wm.save_mainfile(filepath=mainfile_path)

def bclear():
    # remove all elements
    # object mode
    # test if there is any object
    if (len(bpy.data.objects) == 0):
        return

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # set cursor to (0, 0, 0)
    bpy.context.scene.cursor.location = (0, 0, 0)

