import numpy as np 
from dataclasses import dataclass 

@dataclass 
class CameraInfo:
    R: np.array = np.eye(3)
    T: np.array = np.zeros(3)
    FovY: float = 60 / 180 * np.pi
    ResW: int = 400
    ResH: int = 400

    @property 
    def FovX(self):
        return 2 * np.arctan(np.tan(0.5 * self.FovY) * self.ResW / self.ResH)   
    
    def to_dict(self):
        return {
            "R": self.R.flatten().tolist(),
            "T": self.T.flatten().tolist(),
            "FovY": self.FovY,
            "ResW": self.ResW,
            "ResH": self.ResH
        }