import pytest

import torch 
import numpy as np
from PIL import Image

@pytest.mark.current 
def test_intri():
    intr_path = "asset/image/intrinsics.npy"
    intr = torch.from_numpy(np.load(intr_path))
    # float close
    assert intr.shape == torch.Size([3, 3])
    assert intr[0][0] == pytest.approx(518.8579)
    assert intr[0][1] == pytest.approx(0.0) 
    assert intr[0][2] == pytest.approx(325.5824)
    assert intr[1][0] == pytest.approx(0.0)
    assert intr[1][1] == pytest.approx(519.4696)
    assert intr[1][2] == pytest.approx(253.7362)
    assert intr[2][0] == pytest.approx(0.0)
    assert intr[2][1] == pytest.approx(0.0)
    assert intr[2][2] == pytest.approx(1.0)

@pytest.mark.current
def test_img():
    img_path = "asset/image/rgb.png"
    img = np.array(Image.open(img_path))
    # shape
    assert img.shape == (480, 640, 3)
    assert img.dtype == np.uint8a