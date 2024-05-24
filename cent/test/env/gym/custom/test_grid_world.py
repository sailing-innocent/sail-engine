# grid world is a customed env
# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
import numpy as np 
import pygame 
import pytest

import gymnasium as gym
from gymnasium import spaces

class GridWorld(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size 
        self.window_size = 512

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size-1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size-1, shape=(2,), dtype=int),
            }
        )

        # we have four actions: up, down, left, right
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, -1]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        if human mode, 
            `self.window` should be a reference to the window,
            `self.clock` should be a reference to the clock,
        """
        self.window = None
        self.clock = None

    # translate the environment state into an observation
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    # translate the environment state into an auxiliary information
    # info will contain such data only available for step() function
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location,
                ord=1,
            )
        }

    # reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # choose the agent location randomly
        self._agent_location = self.np_random.integers(0, self.size, size=(2,), dtype=int)

        # sample the target location randomly, but not the same as the agent location
        self._target_location = self._agent_location
        while np.all(self._target_location == self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=(2,), dtype=int
            )
        
        # return the observation
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info

    def step(self, action):
        # map the action to direction
        direction = self._action_to_direction[action]

        # move the agent, assure the agent is in the grid world
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # an episode is done when the agent reaches the target
        terminated = np.all(self._agent_location == self._target_location)
        reward = 1.0 if terminated else 0.0 # binary sparse reward
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        # return observation, reward, terminated, truncated, info
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            # init pygame window
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        
        if self.clock is None and self.render_mode == 'human':
            # init pygame clock
            self.clock = pygame.time.Clock()

        # draw the background
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size // self.size # the size of each grid in pixel

        # draw target in red rectangle
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            )
        )

        # draw the agent in blue circle

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            pix_square_size * (self._agent_location + 0.5),
            pix_square_size // 3,
        )

        # some grid lines

        for i in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, i * pix_square_size),
                (self.window_size, i * pix_square_size),
                width=3,
            )
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (i * pix_square_size, 0),
                (i * pix_square_size, self.window_size),
                width=3,
            )
        
        # blit

        if self.render_mode == 'human':
            # update the window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata['render_fps'])
        else: # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2)),
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

@pytest.mark.current
def test_grid_world():
    grid_world = GridWorld(render_mode='human')
    grid_world.reset()
    for _ in range(50):
        obs, reward, terminated, truncated, info = grid_world.step(grid_world.action_space.sample())
        if terminated:
            break
    grid_world.close()
    assert True