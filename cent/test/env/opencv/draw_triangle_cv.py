import cv2 as cv 
import numpy as np

width = 512
height = 512
img = np.ones((width,height,3), np.uint8)
img = 255 * img

lu = [100,100]
ru = [100,200]
lb = [300,100]
rb = [300,200]
# draw a rectangle

def in_rect(_i, _j, _lu, _ru, _lb, _rb):
    return _i > _lu[0] and _i < _lb[0] and _j > _lu[1] and _j < _ru[1]

def in_tri(_i, _j, _a, _b, _c):
    return line_det(_i, _j, _a, _b) > 0 and line_det(_i, _j, _b, _c) > 0 and line_det(_i, _j, _c, _a) > 0
    
def line_det(_i, _j, _a, _b):
    return (_b[1]-_a[1])*(_i-_a[0])-(_b[0]-_a[0])*(_j-_a[1])

a = [200,200]
color_a = np.array([255,0,0])

b = [100,300]
color_b = np.array([0,255,0])

c = [300,400]
color_c = np.array([0,0,255])

def interp(t, a, b):
    return a * t + b * (1 - t)

def interp_color(i,j,a,b,c,color_a, color_b, color_c):
    A = np.array([[b[0]-a[0], b[1]-a[1]],[c[0]-a[0], c[1]-a[1]]])
    x = np.array([i-a[0],j-a[1]])
    mn = np.matmul(x, np.linalg.inv(A)) 
    # print(mn)
    return interp(mn[0], color_a, color_b) + interp(mn[1], color_a, color_c)

for i in range(width):
    for j in range(height):
        if in_tri(i, j, a, b, c):
            color = interp_color(i,j,a,b,c,color_a, color_b, color_c).astype(np.uint8)
            img[i,j] = color

cv.imshow('img', img)

if cv.waitKey(0) == 27:
    cv.destroyAllWindows()