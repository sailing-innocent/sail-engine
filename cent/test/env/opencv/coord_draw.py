import cv2 as cv
import numpy as np 

width = 512
height = 512

coord = np.ones([width, height, 3], dtype=np.uint8)
coord = coord * 255 # plain white

origin = np.array([width/2, height/2])
x_axis_dir = np.array([0,1])
y_axis_dir = np.array([-1,0])

max_coord = 200
x_end = max_coord * x_axis_dir + origin
y_end = max_coord * y_axis_dir + origin

grid_size = 10

max_line_weight = 2
min_line_weight = 0

def abs(x):
    if x >= 0:
        return x
    else:
        return -x

def lerp(t, a, b):
    return a * ( 1 - t ) + b * t 

def inner_product(x, y):
    return x[0] * y[0] + x[1] * y[1]

def len_point_to_line(p, x, y):
    return (abs((y[0]-x[0])*(p[1]-x[1]) - (y[1]-x[1])*(p[0]-x[0]))/np.sqrt((y[0]-x[0])*(y[0]-x[0])+(y[1]-x[1])*(y[1]-x[1])))

def gradient_line_to_draw(i, j, _from, _to, start_weight, end_weight):
    ap = np.array([i - _from[0], j - _from[1]])
    ab = np.array([_to[0] - _from[0], _to[1] - _from[1]])
    ratio = inner_product(ap, ab) / inner_product(ab, ab)
    width = lerp(ratio, start_weight, end_weight)
    if (ratio > 0 and ratio < 1):
        return len_point_to_line(np.array([i,j]), _from, _to) < width - 0.5 
    else:
        return False

def line_to_draw(i, j, _from, _to, weight):
    return gradient_line_to_draw(i, j, _from, _to, weight, weight)

def under_func(i, j, o):
    x = o[1] - j
    y = i - o[0]
    return abs(x * y) < (max_coord * max_coord / 16)

max_step = np.uint(max_coord / grid_size)

def grid_to_draw(i, j, grid_size, max_step, origin, x_dir, y_dir):
    in_grid = False
    for step in range(max_step):
        in_grid = in_grid or (gradient_line_to_draw(i, j, origin + step * grid_size * x_dir, y_end + step * grid_size * x_dir, 1, 0) and under_func(i, j, origin))
        in_grid = in_grid or (gradient_line_to_draw(i, j, origin + step * grid_size * y_dir, x_end + step * grid_size * y_dir, 1, 0) and under_func(i, j, origin))
    return in_grid

for i in range(width):
    for j in range(height):
        if line_to_draw(i, j, origin, x_end, max_line_weight):
            coord[i][j] = np.zeros([3], dtype=np.uint8)
        if line_to_draw(i, j, origin, y_end, max_line_weight):
            coord[i][j] = np.zeros([3], dtype=np.uint8)
        if grid_to_draw(i, j, grid_size, max_step, origin, x_axis_dir, y_axis_dir):
            coord[i][j] = np.zeros([3], dtype=np.uint8)

cv.imshow("coord", coord)

if cv.waitKey(0) == 27:
    cv.destroyAllWindows()