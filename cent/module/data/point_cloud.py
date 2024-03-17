import numpy as np 
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array # N,3
    colors : np.array # N,3
    normals : np.array # N,3