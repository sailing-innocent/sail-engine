import pytest 
import sys 
import os 
cwd = os.path.join(os.getcwd(), "../bin/release")
sys.path.append(cwd)
from innopy import PointRenderApp
import matplotlib.pyplot as plt
import numpy as np 
import torch 

from module.utils.camera.basic import Camera

@pytest.mark.current 
def test_point_render_cuda():
    app = PointRenderApp()
    app.create(cwd, "cuda")
    width = 800
    height = 600
    result_img = torch.zeros((height, width, 3), dtype=torch.float32).cuda()
    cam = Camera()
    cam.lookat(np.array([2, 2, 2]), np.array([0, 0, 0]))
    cam.set_res(width, height)
    # flatten to list
    # numpy is row-major and lc is col-major
    view_matrix_arr = cam.view_matrix.T.flatten().tolist()
    # print(view_matrix_arr)
    proj_matrix_arr = cam.proj_matrix.T.flatten().tolist()

    N = 1000
    # random points in [0, 1]
    xyz = torch.rand((N, 3), dtype=torch.float32).cuda()
    color = torch.ones((N, 3), dtype=torch.float32).cuda()

    app.render_cuda(height, width, result_img.contiguous().data_ptr(), N, xyz.contiguous().data_ptr(), color.contiguous().data_ptr(), view_matrix_arr, proj_matrix_arr)
    result_img_np = result_img.detach().cpu().numpy()
    plt.imshow(result_img_np)
    plt.show()