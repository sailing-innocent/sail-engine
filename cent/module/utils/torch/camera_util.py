# torch based camera utility functions
import torch 

# DK2Points
# -------------------------------------------------------
# @param depth_map: (H, W) depth map
# @param intrinsics: (3, 3) intrinsics matrix
# @param device: device
# @return: (H, W, 3) point cloud
# -------------------------------------------------------
# depth-map D, intrinsics K to point cloud X,Y,Z
def DK2Points(depth_map, intrinsics, device="cuda"):
    # get the shape
    H, W = depth_map.shape
    # get the intrinsics
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    # get the grid
    x = torch.arange(0, W, device=device).float().repeat(H, 1)
    y = torch.arange(0, H, device=device).float().repeat(W, 1).t()
    # get the points
    X = (x - cx) * depth_map / fx
    Y = (y - cy) * depth_map / fy
    Z = depth_map
    # stack
    return torch.stack([X, Y, Z], dim=2)