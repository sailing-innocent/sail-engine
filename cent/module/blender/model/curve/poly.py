import bpy 

import numpy as np 
from typing import NamedTuple

class ControlPoint(NamedTuple):
    co: np.ndarray
    radius: float = 0.1

class Spline:
    def __init__(self, spline_type="POLY"):
        self.points = []
        self.radius_list = None 
        self.radius = 0.1
        self.spline_type = spline_type
    
    def add_point(self, point: ControlPoint):
        self.points.append(point)

class Curve:
    def __init__(self, name="dummy_curve", curve_type="CURVE"):
        self.name = name 
        self.curve_type = curve_type
        self.dimensions = '3D'
        self.bevel_depth = 1
        self.resolution_u = 4 
        self.fill_mode = 'FULL'
        self.splines = []

    def add_spline(self, spline: Spline):
        self.splines.append(spline)

    def create_obj(self):
        curve = bpy.data.curves.new(name=self.name, type=self.curve_type)
        curve.dimensions = self.dimensions
        curve.bevel_depth = self.bevel_depth
        curve.resolution_u = self.resolution_u
        curve.fill_mode = self.fill_mode

        for spline in self.splines:
            spline_obj = curve.splines.new(type=spline.spline_type)
            spline_obj.points.add(len(spline.points)-1)
            for i, point in enumerate(spline.points):
                spline_obj.points[i].co = (point.co[0], point.co[1], point.co[2], 1)
                spline_obj.points[i].radius = point.radius

        curve_obj = bpy.data.objects.new(self.name, curve)
        bpy.context.collection.objects.link(curve_obj)
        return curve_obj
    