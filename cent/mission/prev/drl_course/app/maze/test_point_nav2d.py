import pytest 

import numpy as np 

from scene.grid.point_nav_2d import PointNav2D

@pytest.mark.app
def test_point_nav_2d():
    point_nav_2d = PointNav2D(
        render_mode='human'
    )

    point_nav_2d.reset()

    for _ in range(5):
        obs, reward, terminated, truncated, info = point_nav_2d.step(
            point_nav_2d.action_space.sample() # random sample
        )

        if terminated:
            break
  
    point_nav_2d.close()
    assert True