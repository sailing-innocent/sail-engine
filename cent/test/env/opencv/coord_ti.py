import cv2 as cv
import numpy as np 
import taichi as ti

ti.init(arch=ti.gpu)

img = cv.imread("D:/data/images/river_001_1920x1080.jpg")
SEG_PIXELS = 50
UPSIZE_RATIO = 10
img_seg = img[0:SEG_PIXELS,0:SEG_PIXELS]
upsized_img = np.ones([SEG_PIXELS*UPSIZE_RATIO, SEG_PIXELS*UPSIZE_RATIO, 3], dtype=np.float64)
for i in range(SEG_PIXELS):
    for j in range(SEG_PIXELS):
        for itx in range(UPSIZE_RATIO):
            for ity in range(UPSIZE_RATIO):
                upsized_img[i * UPSIZE_RATIO +itx][j * UPSIZE_RATIO+ity] = img_seg[i][j]

width = 1920
height = 1080

coord = np.ones([width, height, 3], dtype=np.float64)
coord = coord * 255 # plain white

origin = np.array([width/2, height/2])
origin_ti = ti.field(ti.i32,shape=[2])
origin_ti.from_numpy(origin)

x_axis_dir = np.array([0,1])
y_axis_dir = np.array([-1,0])

x_dir_ti = ti.field(ti.i32,shape=[2])
x_dir_ti.from_numpy(x_axis_dir)
y_dir_ti = ti.field(ti.i32,shape=[2])
y_dir_ti.from_numpy(y_axis_dir)

max_coord = 200
x_end = max_coord * x_axis_dir + origin
y_end = max_coord * y_axis_dir + origin

x_end_ti = ti.field(ti.i32,shape=[2])
x_end_ti.from_numpy(x_end)

y_end_ti = ti.field(ti.i32,shape=[2])
y_end_ti.from_numpy(y_end)

grid_size = 10

max_line_weight = 2
min_line_weight = 0

@ti.func
def abs(x):
    if x < 0:
        x = -x
    return x

@ti.func
def lerp(t, a, b):
    return a * ( 1 - t ) + b * t 

@ti.func
def inner_product(x, y):
    return x[0] * y[0] + x[1] * y[1]

@ti.func
def len_point_to_line(p, x, y):
    return (abs((y[0]-x[0])*(p[1]-x[1]) - (y[1]-x[1])*(p[0]-x[0]))/ti.sqrt((y[0]-x[0])*(y[0]-x[0])+(y[1]-x[1])*(y[1]-x[1])))

@ti.func
def gradient_line_to_draw(i, j, _from, _to, start_weight, end_weight):
    ap = ti.Vector([i - _from[0], j - _from[1]])
    ab = ti.Vector([_to[0] - _from[0], _to[1] - _from[1]])
    ratio = inner_product(ap, ab) / inner_product(ab, ab)
    width = lerp(ratio, start_weight, end_weight)
    flag = False
    if (ratio > 0 and ratio < 1):
        flag = len_point_to_line(ti.Vector([i,j]), _from, _to) < width - 0.5 
    
    return flag

@ti.func
def line_to_draw(i, j, _from, _to, weight):
    return gradient_line_to_draw(i, j, _from, _to, weight, weight)

@ti.func
def under_func(i, j, o):
    x = o[1] - j
    y = i - o[0]
    return abs(x * y) < (max_coord * max_coord / 16)

max_step = np.uint(max_coord / grid_size)

## INIT GRID TERMINALS GROUP
grid_ti = ti.Vector.field(2, ti.i32, shape=(2, max_step, 2))
for step in range(max_step):
    for idx in range(2):
        grid_ti[0,step,0][idx] = origin_ti[idx] + step * grid_size * x_dir_ti[idx]
        grid_ti[0,step,1][idx] = y_end_ti[idx] + step * grid_size * x_dir_ti[idx]
        grid_ti[1,step,0][idx] = origin_ti[idx] + step * grid_size * y_dir_ti[idx]
        grid_ti[1,step,1][idx] = x_end_ti[idx] + step * grid_size * y_dir_ti[idx]

@ti.func
def grid_to_draw(i, j, max_step, grid):
    in_grid = False
    for step in range(max_step):  
        if (gradient_line_to_draw(i, j, grid[0, step, 0], grid[0, step, 1], 1, 0.5)):
            in_grid = True
        if (gradient_line_to_draw(i, j, grid[1, step, 0], grid[1, step, 1], 1, 0.5)):
            in_grid = True
    return in_grid

coord_ti = ti.Vector.field(3, ti.f64, shape=(width, height))
coord_ti.from_numpy(coord)

@ti.func
def circle_to_draw(i, j, centx, centy, r, width):
    return abs((i - centx) * (i - centx) + (j - centy) * (j - centy) - r * r) < width * width


@ti.kernel
def draw(t: int):
    black = ti.Vector([0,0,0], ti.f64)
    for i, j in coord_ti:
        if line_to_draw(i, j, origin_ti, x_end_ti, max_line_weight):
            coord_ti[i,j] = black
        if line_to_draw(i, j, origin_ti, y_end_ti, max_line_weight):
            coord_ti[i,j] = black
        if grid_to_draw(i, j, max_step, grid_ti) and under_func(i, j, origin_ti):
            coord_ti[i,j] = black

        """
        if circle_to_draw(i, j, 100, 200, 50, 8):
            coord_ti[i,j] = black
        """

draw(0)
coord = coord_ti.to_numpy()

"""
cv.imshow("coord", coord)
cv.imshow("imgseg", upsized_img)

if cv.waitKey(0) == 27:
    cv.destroyAllWindows()

"""

cv.imwrite("coord.jpg", coord)