import pytest 
import taichi as ti 

@pytest.mark.func 
def test_shape():
    ti.init()
    fd = ti.field(dtype=ti.f32,shape=[2,2])
    assert len(fd.shape) == 2 # dimension
    assert fd.shape[0] == 2 # dim 1 size
    assert fd.shape[1] == 2 # dim 2 size

