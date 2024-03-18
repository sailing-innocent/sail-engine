import numpy as np

def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))