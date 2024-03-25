import pytest 
import torch 
import numpy as np 
import cv2 as cv 
from app.renderer.ray_marcher.torch.marcher import RayMarcherModule


@pytest.mark.current 
def test_ray_march_torch():
    W = 512
    H = 512 
    i, j = torch.meshgrid(torch.linspace(0, 1, W), torch.linspace(0, 1, H), indexing='ij')
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-0.5)/1, -(j-0.5)/1, -torch.ones_like(i)], -1)
    obj = torch.Tensor([[0.0, 0.0, 0.0, 1.0]])
    dirs = torch.nn.functional.normalize(dirs, dim=-1)
    marcher = RayMarcherModule()
    pixels = marcher(obj, dirs)
    pixels = (pixels.numpy() * 255.0).astype(np.uint8)
    cv.imshow("pixels", pixels)
    k = cv.waitKey(0)
    assert True 