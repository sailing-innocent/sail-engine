import pytest
from kornia import create_meshgrid 

@pytest.mark.func
def test_create_meshgrid():
    # create_meshgrid(H, W, normilized_coord, device, dtpye)
    grid = create_meshgrid(2, 3, False, device='cpu')[0]  # (H, W, 2)
    assert grid.shape == (2,3,2)
    assert grid[0,0,0] == 0
    assert grid[0,0,1] == 0
    assert grid[0,1,0] == 1
    assert grid[0,1,1] == 0
    assert grid[0,2,0] == 2
    assert grid[0,2,1] == 0
    assert grid[1,0,0] == 0
    assert grid[1,0,1] == 1
    assert grid[1,1,0] == 1
    assert grid[1,1,1] == 1
    assert grid[1,2,0] == 2
    assert grid[1,2,1] == 1
