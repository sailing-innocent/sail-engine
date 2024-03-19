import pytest

import numpy as np

from scene.grid.maze import generate_maze, print_maze, get_maze_debug
from scene.grid.maze_world import MazeWorld
from scene.grid.planning import bfs_policy

@pytest.mark.app
def test_maze_test():
    maze = generate_maze()
    maze_debug_str = get_maze_debug(maze)
    print(maze_debug_str)
    assert True 

@pytest.mark.app
def test_maze_dfs():
    maze = generate_maze(
        mode='dfs', 
        width=8, 
        height=8, 
        startnode=(0, 0), endnode=(7, 7)
    )
    print_maze(maze)
    assert True 

@pytest.mark.app
def test_maze_world():
    maze_world = MazeWorld(
        render_mode='human',
        gen_mode='dfs',
        size=8,
        startnode=(0, 0), endnode=(7, 7)
    )
    maze_world.reset()
    print(maze_world.debug())
    for _ in range(50):
        obs, reward, terminated, truncated, info = maze_world.step(
            maze_world.action_space.sample() # random sample
        )
        if terminated:
            break

    maze_world.close()
    assert True 

@pytest.mark.app
def test_maze_world_bfs():
    maze_world = MazeWorld(
        render_mode='human',
        gen_mode='dfs',
        size=8,
        startnode=(0, 0), endnode=(7, 7)
    )
    maze_world.reset()
    # print(maze_world.debug())
    action_list = bfs_policy(maze_world.maze, maze_world.startnode, maze_world.endnode)
    # print(action_list)
    for action in action_list:
        obs, reward, terminated, truncated, info = maze_world.step(action)
        if terminated:
            break

    maze_world.close()
    assert True 
