import bpy 
from app.renderer.dummy import DummyRenderer

class DummyRenderEngine(bpy.types.RenderEngine):
    bl_idname = "DUMMY"
    bl_label = "Dummy"
    bl_use_preview = False 

    def __init__(self):
        self.scene_data = None
        self.draw_data = None

    def __del__(self):
        pass 

    def render(self, depsgraph):
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        self.size_x = int(scene.render.resolution_x * scale)
        self.size_y = int(scene.render.resolution_y * scale)
        renderer = DummyRenderer(self.size_x, self.size_y)
        pixels = renderer.render()
        rect = [pixels[i, j, :].tolist() for i in range(self.size_x) for j in range(self.size_y)]
        # Here we write the pixel values to the RenderResult
        result = self.begin_result(0, 0, self.size_x, self.size_y)
        layer = result.layers[0].passes["Combined"]
        layer.rect = rect
        self.end_result(result)

    def view_update(self, context, depsgraph):
        pass 

    def view_draw(self, context, depsgraph):
        pass 