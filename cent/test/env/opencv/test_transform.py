import pytest 
import cv2 as cv
import numpy as np

@pytest.mark.current 
def test_rotation():
    R = cv.Rodrigues(np.array([0,np.pi/2,0], dtype=np.float32))
    print(R[0])