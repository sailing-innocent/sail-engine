import numpy as np 
from typing import NamedTuple



class BasicPointCloud(NamedTuple):
    points : np.array # N,3
    colors : np.array # N,3
    normals : np.array # N,3

def sphere_point_cloud(r = 1.0, N = 1000, color=[1,0,0]):
    np.random.seed(0)
    theta = np.random.uniform(0, 2*np.pi, N)
    phi = np.random.uniform(0, np.pi, N)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    points = np.stack([x,y,z], axis = 1)
    
    colors = np.array([color] * N)
    normals = points
    return BasicPointCloud(points, colors, normals)