import pytest 
import gymnasium as gym

@pytest.mark.app
def test_cartpole():
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset(seed=42)
    assert observation.shape == (4,)
    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    assert True