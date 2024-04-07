import numpy as np 
if __name__ == "__main__":
    depth_f = "D:/dataset/mip360d/bicycle/depth_4/_DSC8679.npy"
    depth = np.load(depth_f)
    print(depth)