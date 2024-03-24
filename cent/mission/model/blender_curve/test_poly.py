import pytest 
from module.blender.wrapper import blender_executive 
from module.blender.model.curve.poly import ControlPoint, Spline, Curve

@blender_executive
def blender_poly_suite(rootdir):
    point00 = ControlPoint(co=[0, 0, 0])
    point01 = ControlPoint(co=[1, 0, 0])
    point02 = ControlPoint(co=[1, 1, 0])

    spline0 = Spline()
    spline0.add_point(point00)
    spline0.add_point(point01)
    spline0.add_point(point02)

    curve = Curve()
    curve.add_spline(spline0)
    curve.create_obj()

@pytest.mark.current 
def test_poly():
    blender_poly_suite(subfolder="model", filename="test_durve_poly")
    assert True