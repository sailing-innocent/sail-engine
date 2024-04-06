import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# read a depth image
# get its intrinsics\
# return its point cloud
import torch 
import numpy as np
from PIL import Image

from module.utils.torch.camera_util import DK2Points
from module.utils.pointcloud.io import storePly

if __name__ == "__main__":
    intr_path = "asset/image/intrinsics.npy"
    intr = torch.from_numpy(np.load(intr_path))
    assert intr.shape == torch.Size([3, 3])
    depth_img_path = "asset/image/depth.png"
    depth_img = np.array(Image.open(depth_img_path))
    scales = 1000.0
    
    # shape
    assert depth_img.shape == (480, 640)
    assert depth_img.dtype == np.int32

    H = depth_img.shape[0]
    W = depth_img.shape[1]
    img_torch = torch.from_numpy(depth_img).float().reshape(H, W)
    # print(img_torch)

    points = DK2Points(img_torch, intr, "cpu")
    points_np = points.reshape(-1, 3).detach().cpu().numpy() / scales
    print(points_np.shape)
    print(points_np[points_np > 0])

    color_img_path = "asset/image/rgb.png"
    color = np.array(Image.open(color_img_path)) / 255.0
    color = color.reshape(-1, 3)
    ply_path = "D:/temp/depth.ply"
    storePly(ply_path, points_np, color)