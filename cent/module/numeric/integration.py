import numpy as np 

def riemann_sum_np(f, x_start, x_end, n=100):
    delta_x = (x_end - x_start) / n 
    rsum = 0
    for i in range(1, n+1):
        xi = np.pi + np.pi / n * i 
        rsum = rsum + f(xi) * delta_x
    return rsum

