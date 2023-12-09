import pytest 

from lib.reimpl.svox2 import SparseGrid, Camera, Rays, RenderOptions

@pytest.mark.current 
def test_svox2():
    ckpt = "E:/pretrained/plenoxel/nerf_blender_chair.npz"
    device = "cuda:0"
    grid = SparseGrid.load(ckpt, device=device)
    assert grid.use_background == False 
