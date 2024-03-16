from dataclasses import dataclass 
import numpy as np 

@dataclass
class SphereInfo:
    center: np.ndarray
    radius: float