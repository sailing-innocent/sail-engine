import pytest
from app.renderer.ray_marcher.taichi.marcher import TaichiRayMarcher

@pytest.mark.current
def test_ti_ray_marcher():
    marcher = TaichiRayMarcher()
    marcher.run()
    assert True 
