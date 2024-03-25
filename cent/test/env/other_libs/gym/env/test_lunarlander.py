import pytest 
import gymnasium as gym

@pytest.mark.app
def test_lunarlander():
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset() # get the first observation

    for _ in range(100):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action) # the environment update according to agent's action

        if terminated or truncated:
            observation, info = env.reset()

    env.close()