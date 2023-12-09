import torch 
#  -------------------------------------------------------
#  qvec2R - convert quaternion to rotation matrix
#  qvec: (4,) quaternion
#  device: device
#  qtype: 'rxyz' or 'xyzr'
#  return: (N, 3, 3) rotation matrix
#  R = [[1 - 2*(y^2 + z^2), 2*(x*y - r*z), 2*(x*z + r*y)],
#       [2*(x*y + r*z), 1 - 2*(x^2 + z^2), 2*(y*z - r*x)],
#       [2*(x*z - r*y), 2*(y*z + r*x), 1 - 2*(x^2 + y^2)]]
#  -------------------------------------------------------
def qvec2R(qvec, device="cuda", qtype='rxyz'):
    norm = torch.sqrt(qvec[:,0]*qvec[:,0] + qvec[:,1]*qvec[:,1] + qvec[:,2]*qvec[:,2] + qvec[:,3]*qvec[:,3])
    q = qvec / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=device)
    if (qtype=='rxyz'):
        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]
    elif (qtype=='xyzr'):
        x = q[:, 0]
        y = q[:, 1]
        z = q[:, 2]
        r = q[:, 3]
    else:
        raise ValueError(f"qtype {qtype} not supported")

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    return R

def T2Sigma(s1,s2,theta):
    c = torch.cos(theta)
    s = torch.sin(theta)
    c00 =  c * c * s1 * s1 + s * s * s2 * s2
    c01 =  (s1 * s1 - s2 * s2) * c * s
    c11 =  c * c * s2 * s2 + s * s * s1 * s1
    # -> N x 3
    return torch.stack([c00, c01, c11], dim=1)

