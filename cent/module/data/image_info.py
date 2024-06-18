import numpy as np 

class ImageInfo:
    data: np.array # HWC data
    W: int
    H: int
    C: int 
    ydown: bool = True

    def __init__(self, data, W, H, C):
        self.data = data
        self.W = W
        self.H = H
        self.C = C