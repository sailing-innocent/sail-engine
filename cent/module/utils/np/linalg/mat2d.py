import numpy as np 

def sym2d_to_scale_theta(mat2d: np.ndarray):
    # calculate scale
    a = mat2d[..., 0, 0]
    b = mat2d[..., 0, 1]
    c = mat2d[..., 1, 1]

    det = np.sqrt(4 * b * b + (c - a) * (c - a))
    s1 = 0.5 * (a + c + det)
    s2 = 0.5 * (a + c - det)

    theta = np.arctan(2 * b / (a-c)) / 2
    return s1, s2, theta

def scale_theta_to_sym2d(s1, s2, theta):
    S = scale_to_S(s1, s2)
    R = theta_to_R(theta)
    # bmm RSR^T
    return R @ S @ R.T

def theta_to_R(theta):
    """Converts rotation angle to 2D rotation matrix.

    Args:
        theta (float): Rotation angle.

    Returns:
        np.ndarray: 2D rotation matrix.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    col_1 = np.stack([cos_theta, sin_theta], axis=-1)
    col_2 = np.stack([-sin_theta, cos_theta], axis=-1)
    return np.stack([col_1, col_2], axis=-1)

def scale_to_S(s1, s2):
    """Converts scale to 2D scale matrix.

    Args:
        s1 (float): Scale 1.
        s2 (float): Scale 2.

    Returns:
        np.ndarray: 2D scale matrix.
    """
    row_1 = np.stack([s1, np.zeros_like(s1)], axis=-1)
    row_2 = np.stack([np.zeros_like(s2), s2], axis=-1)
    return np.stack([row_1, row_2], axis=-1)