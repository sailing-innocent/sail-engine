import torch 

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

#  -------------------------------------------------------
#  qvec_R - convert quaternion to rotation matrix
#  qvec: (4,) quaternion
#  device: device
#  qtype: 'rxyz' or 'xyzr'
#  return: (N, 3, 3) rotation matrix
#  R = [[1 - 2*(y^2 + z^2), 2*(x*y - r*z), 2*(x*z + r*y)],
#       [2*(x*y + r*z), 1 - 2*(x^2 + z^2), 2*(y*z - r*x)],
#       [2*(x*z - r*y), 2*(y*z + r*x), 1 - 2*(x^2 + y^2)]]
#  -------------------------------------------------------
def qvec_R(qvec, device="cuda", qtype='rxyz'):
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

# N, 3, 3 -> N, 6 for fetching lower diagonal
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

# N,3,3 -> N,6 for compressing symmetric matrix
def strip_symmetric(sym):
    return strip_lowerdiag(sym)
