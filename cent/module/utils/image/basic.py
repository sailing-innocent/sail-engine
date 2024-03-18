from module.data.image_info import ImageInfo
import numpy as np 
import matplotlib.pyplot as plt 

class Image:
    def __init__(self,
        W=800,H=600,C=3,color=[0.5,0.5,0.5]):
        self.info = ImageInfo(
            data = np.repeat(color, W * H).reshape(H, W, C),
            W = W,
            H = H,
            C = C
        )
        self.data_type = "FLOAT32"

    def load_from_info(self, info: ImageInfo):
        self.info = info
        self.data_type = "FLOAT32"
        if (self.info.data.dtype == np.uint8):
            self.data_type = "UINT8"

    def load_from_file(self, path: str):
        self.info.data = np.array(plt.imread(path))
        self.data_type = "FLOAT32"
        if (self.info.data.dtype == np.uint8):
            self.data_type = "UINT8"
        self.info.W = self.info.data.shape[1]
        self.info.H = self.info.data.shape[0]
        self.info.C = self.info.data.shape[2]

    def load_from_data(self, _data: np.array):
        self.info.data = _data
        self.info.W = _data.shape[1]
        self.info.H = _data.shape[0]
        self.info.C = _data.shape[2]

    def save(self, path: str):
        self.to_uint8()
        plt.imsave(path, self.info.data)
    
    def to_uint8(self):
        if (self.data_type == "FLOAT32"):
            self.info.data = (self.info.data * 255.0).astype(np.uint8)
            self.data_type = "UINT8"

    def to_float32(self):
        if (self.data_type == "UINT8"):
            self.info.data = self.info.data.astype(np.float32) / 255.0
            self.data_type = "FLOAT32"

    @property
    def W(self):
        return self.info.W
    
    @property
    def H(self):
        return self.info.H
    
    @property
    def C(self):
        return self.info.C
    
    @property 
    def shape(self):
        return self.info.data.shape

    def blend(self, color: np.array):
        assert color.shape == (3,)
        assert self.C == 4
        alpha = self.info.data[:, :, 3:]
        self.info.data = (self.info.data[:, :, :3] * alpha) + (color * (1 - alpha))

    def flip_y(self):
        np.flip(self.info.data, axis=0)
        self.info.ydown = not self.info.ydown

    def show(self, immediate = False):
        plt.imshow(self.info.data)
        if immediate:
            plt.show() 