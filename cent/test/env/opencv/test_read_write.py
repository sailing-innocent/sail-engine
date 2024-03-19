import pytest
import cv2 as cv 
import sys 

import torch
import numpy as np 

@pytest.mark.app
def test_cv2_io():
    img = cv.imread("assets/image/avatar_60x60.png")
    if img is None:
        sys.exit("Could not read the image")

    assert img.shape == (61, 60, 3)

    cv.imshow("Display window", img)
    k = cv.waitKey(0)

    if k == ord("s"):
        cv.imwrite("avatar60x60.png", img)
    
    assert True 

@pytest.mark.app
def test_torch_to_img():
    half_img_l = torch.zeros([256,512,3],dtype=torch.uint8)
    half_img_r = torch.ones([256,512,3],dtype=torch.uint8)
    img = torch.cat((half_img_l, half_img_r))
    assert img.shape == (512, 512, 3)
    img = img * 255
    img = img.numpy()
    assert img.shape == (512,512,3)
    assert img.dtype == np.uint8
    cv.imshow("upside black and downside white", img)
    k = cv.waitKey(0)
    # upside black and downside white
    assert True