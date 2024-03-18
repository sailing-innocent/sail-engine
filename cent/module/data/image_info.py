from dataclasses import dataclass 
import numpy as np 
@dataclass
class ImageInfo:
    data: np.array # HWC data
    W: int
    H: int
    C: int 
    ydown: bool = True