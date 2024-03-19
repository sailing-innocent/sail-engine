import numpy as np 

class DummyRenderer:
    def __init__(self, w = 256, h = 256):
        self.w = w 
        self.h = h
    def render(self):
        return np.ones((self.w, self.h, 4))