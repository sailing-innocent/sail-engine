import numpy as np 
import sys 
import os 
cwd = os.path.join(os.getcwd(), "../bin/release")
sys.path.append(cwd)
from innopy import DummyRenderApp

class DummyRenderer:
    def __init__(self):
        self.app = DummyRenderApp()
        self.app.create(cwd, "cuda")

    def render(self, w = 256, h = 256):
        result_img = self.app.render(h, w)
        result_img_np = np.array(result_img)
        result_img_np = result_img_np.reshape((w, h, 4))
        return result_img_np 