# Author: sailing-innocent
# Date: 2023-05-04
# Brief: Breadth-first search algorithm implementation

import pytest 

import numpy as np 

from scene.grid.maze import print_maze, generate_maze
from scene.grid.planning import bfs, astar

@pytest.mark.app
def test_bfs():
    MAZE_HEIGHT = 10
    MAZE_WIDTH = 10
    start = (0, 0)
    end = (9, 9)
    maze = generate_maze('dfs', MAZE_HEIGHT, MAZE_WIDTH, start, end)
    # maze = generate_maze()
    print_maze(maze)

    traj = bfs(maze, start, end)
    print(traj)
