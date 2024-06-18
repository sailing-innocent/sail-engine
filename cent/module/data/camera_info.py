import numpy as np 

class CameraInfo:
    R: np.array = np.eye(3)
    T: np.array = np.zeros(3)
    FovY: float = 0.6911112070083618
    FovX: float = 0.6911112070083618
    ResW: int = 800
    ResH: int = 800

    def __init__(self, R, T, FovY, FovX, ResW, ResH):
        self.R = R
        self.T = T
        self.FovY = FovY
        self.FovX = FovX
        self.ResW = ResW
        self.ResH = ResH

    def to_dict(self):
        return {
            "R": self.R.flatten().tolist(),
            "T": self.T.flatten().tolist(),
            "FovY": self.FovY,
            "ResW": self.ResW,
            "ResH": self.ResH
        }
    
    @property 
    def K(self):
        return np.array([
            [self.ResW / np.tan(self.FovX / 2) / 2, 0, self.ResW/2],
            [0, self.ResH / np.tan(self.FovY / 2) / 2, self.ResH/2],
            [0, 0, 1]
        ])