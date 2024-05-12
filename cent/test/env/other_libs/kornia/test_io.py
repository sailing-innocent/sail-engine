import pytest
import cv2
from matplotlib import pyplot as plt 
import numpy as np

import torch 
import torchvision
import kornia as K 

@pytest.mark.func
def test_convert_image():
    image_path = "D:/workspace/research/arturito.jpg"
    img_bgr: np.array = cv2.imread(image_path)
    x_bgr: torch.tensor = K.image_to_tensor(img_bgr) # CxHxW / torch.uint8
    x_bgr = x_bgr.unsqueeze(0) # 1xCxHxW

    print(f"convert from '{img_bgr.shape}' to '{x_bgr.shape}'") # from (144,256,3) to torch.Size([1,3,144,256])

    assert img_bgr.shape == (144,256,3)
    assert x_bgr.shape == (1,3,144,256)

@pytest.mark.use
def test_debug_with_matplotlib():
    image_path = "D:/workspace/research/arturito.jpg"
    img_bgr: np.array = cv2.imread(image_path)
    x_bgr: torch.tensor = K.image_to_tensor(img_bgr) # CxHxW / torch.uint8
    x_bgr = x_bgr.unsqueeze(0) # 1xCxHxW

    # convert BGR to RGB
    x_rgb: torch.tensor = K.color.bgr_to_rgb(x_bgr) # 1xCxHxW / torch.uint8

    # convert torch to numpy
    img_bgr: np.array = K.tensor_to_image(x_bgr)
    img_rgb: np.array = K.tensor_to_image(x_rgb)

    # Visualize with Matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(32, 16))
    axs = axs.ravel()

    axs[0].axis("off")
    axs[0].imshow(img_rgb)

    axs[1].axis("off")
    axs[1].imshow(img_bgr)

    plt.show()
    assert 0 == 0