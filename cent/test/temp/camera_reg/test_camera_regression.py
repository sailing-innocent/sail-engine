# use camera as regression
import pytest 
import torch 
from module.utils.camera.basic import Camera
import numpy as np 

@pytest.mark.current 
def test_cam_reg():
    N_points = 10
    N_in_dim = 3
    N_out_dim = 1
    N_sample = 4
    points = torch.randn(N_points, N_in_dim)
    print(points)
    cam01 = Camera("FlipY")
    cam01.lookat(np.array([0, -1, 0]), np.array([0, 0, 0]))
    v1 = torch.from_numpy(cam01.view_matrix).float()
    print(v1)

    cam02 = Camera("FlipY")
    cam02.lookat(np.array([1, 1, 1]), np.array([0, 0, 0]))
    v2 = torch.from_numpy(cam02.view_matrix).float()

    p_hom = torch.cat([points, torch.ones(N_points, 1)], dim=1)
    p_view_1_hom = torch.matmul(p_hom, v1.t())
    p_view_2_hom = torch.matmul(p_hom, v2.t())
    p_view_1 = p_view_1_hom[:, :3] / p_view_1_hom[:, 3].unsqueeze(1)
    p_view_2 = p_view_2_hom[:, :3] / p_view_2_hom[:, 3].unsqueeze(1)
    print(p_view_1)
    p_proj_1 = p_view_1[:, :2] / p_view_1[:, 2].unsqueeze(1)
    print(p_proj_1)
    p_proj_2 = p_view_2[:, :2] / p_view_2[:, 2].unsqueeze(1)

    p_proj = torch.cat([p_proj_1, p_proj_2], dim=1)
    assert p_proj.shape == (N_points, N_sample)
    p_proj = p_proj.reshape(N_points, N_sample, N_out_dim)
    
    # V[2]y - V[0]
    # V[2]y - V[1]

    v01 = torch.cat([v1[0:1, :], v1[1:2,:], v2[0:1, :], v2[1:2, :]], dim = 0)
    v01 = v01.unsqueeze(0).repeat(N_points, 1, 1)
    v2 = torch.cat([v1[2:3, :], v1[2:3, :], v2[2:3, :], v2[2:3, :]], dim = 0)
    v2 = v2.unsqueeze(0).repeat(N_points, 1, 1)

    A = v01 -  v2 * p_proj
    print(A.shape) # 10, 4, 4
    X = A[:, :, 0:3]
    Y = -A[:, :, 3:4]
    assert X.shape == (N_points, N_sample, N_in_dim)
    assert Y.shape == (N_points, N_sample, N_out_dim)
    XTX = torch.matmul(X.transpose(1, 2), X)
    XTX_inv = XTX.inverse()
    XTY = torch.matmul(X.transpose(1, 2), Y)
    W_next = torch.matmul(XTX_inv, XTY)
    print(W_next)