import bpy 
import numpy as np 
import os 

from app.renderer.dummy.d6_impl import DummyRenderer
from app.scene.render.scene import Scene 

class D6ToyEngine(bpy.types.RenderEngine):
    bl_idname = "D6_TOY_ENGINE"
    bl_label = "D6 Toy Engine"
    bl_use_preview = False

    def __init__(self):
        self.scene_data = None 
        self.draw_data = None 
    
    def render(self, depsgraph):
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        self.size_x = int(scene.render.resolution_x * scale)
        self.size_y = int(scene.render.resolution_y * scale)
        
        temp_dir = os.path.join(os.path.dirname(scene.render.filepath), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        # load scene to temp_dir 

        scene_to_render = Scene()
        scene_to_render.from_blender(scene)
        scene_to_render.to_byte_file(temp_dir)

        # render scene

        # renderer = DummyRenderer()
        # pixels = renderer.render(self.size_x, self.size_y)
        # rect = [pixels[i, j, :].tolist() for i in range(self.size_x) for j in range(self.size_y)]
        # result = self.begin_result(0, 0, self.size_x, self.size_y)
        # layer = result.layers[0].passes["Combined"]
        # layer.rect = rect 
        # self.end_result(result)

    def __del__(self):
        pass 
    def view_update(self, context, depsgraph):
        pass 
    def view_draw(self, context, depsgraph):
        pass 