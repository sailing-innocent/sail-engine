import taichi as ti 
import argparse
import math 
import numpy as np 
import cv2 
import os 
import matplotlib.pyplot as plt 

real = ti.f32
ti.init(default_fp=real, arch=ti.cuda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=20)
    options = parser.parse_args()

    print(optinos.iters)

if __name__ == '__main__':
    main()