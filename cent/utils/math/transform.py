import numpy as np 


"""
Convert quaternion to rotation matrix
"""
def qvec2R_np(qvec: np.ndarray) -> np.ndarray:
    # qvec = [w, x, y, z]
    w, x, y, z = qvec
    R = np.array([[1-2*(y**2+z**2), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x**2+z**2), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x**2+y**2)]])
    return R
