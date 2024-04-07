# the basic camera type with numpy implement
from module.data.camera_info import CameraInfo
import numpy as np 

class Camera:
    def __init__(self, catetory: str = "FlipZ"):
        self.info = CameraInfo()
        self.category = catetory
        self.up = np.array([0, 0, 1])

    def from_info(self, info: CameraInfo, category: str = "FlipZ"):
        self.info = info
        self.category = category

    def flip(self):
        self.category = "FlipY" if self.category == "FlipZ" else "FlipZ"
        # rot 180 degree around x axis
        R = np.eye(3)
        R[1, 1] = -1
        R[2, 2] = -1
        self.info.R = np.matmul(self.info.R, R.T)

    def set_res(self, resw: int, resh: int):
        self.info.ResW = resw
        self.info.ResH = resh

    def aspect(self):
        return self.info.ResW / self.info.ResH

    def lookat(self, pos: np.array, target: np.array):
        if self.category == "FlipZ":
            z = pos - target
        elif self.category == "FlipY":
            z = target - pos
        else:
            raise NotImplementedError
        
        x = np.cross(self.up, z)
        # if self.category == "FlipY":
        #     x = -x
        y = np.cross(z, x)
        # noramalize
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = z / np.linalg.norm(z)
        R = np.stack([x, y, z], axis=-1)
        self.info.R = R
        self.info.T = pos

    def world_to_view(self):
        return np.linalg.inv(self.view_to_world())
    
    def view_to_world(self):
        R = self.info.R
        T = self.info.T
        V = np.eye(4)
        V[:3, :3] = R
        V[:3, 3] = T
        return V
    
    @property 
    def view_matrix(self):
        return self.world_to_view()
    
    @property 
    def proj_matrix(self):
        tanfovx = np.tan(0.5 * self.info.FovX)
        tanfovy = np.tan(0.5 * self.info.FovY)
        znear = 0.01
        zfar = 100
        top = tanfovy * znear
        bottom = -top
        right = tanfovx * znear
        left = -right
        P = np.zeros((4, 4))
        z_sign = 1.0
        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -z_sign * zfar * znear / (zfar - znear)
        return P   
    
    @property 
    def full_proj_matrix(self):
        return self.proj_matrix @ self.view_matrix
    
    @property 
    def rays(self):
        i, j = np.meshgrid(
            np.arange(self.info.ResW), 
            np.arange(self.info.ResH), 
            indexing="xy"
        )
        surface_x = 2 * 1.0 * np.tan(0.5 * self.info.FovX)
        surface_y = 2 * 1.0 * np.tan(0.5 * self.info.FovY)
        dirs = np.stack([
            (i - 0.5 * self.info.ResW) / self.info.ResW * surface_x,
            (0.5 * self.info.ResH - j) / self.info.ResH * surface_y,
            - np.ones_like(i)
        ], axis=-1)
        if (self.category == "FlipY"):
            dirs[:, :, 0] *= -1
            dirs[:, :, 2] *= -1

        dirs = np.matmul(dirs, self.info.R.T)
        orig = self.info.T

        return orig, dirs