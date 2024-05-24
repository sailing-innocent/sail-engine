import gymnasium as gym
import pytest 

@pytest.mark.app 
def test_cartpole():
    env = gym.make("CartPole-v1")
    # env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset(seed=42)
    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()

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

@pytest.mark.app
def test_mountain_car():
    env = gym.make("MountainCar-v0", render_mode="human")
    assert env.observation_space.shape == (2,)
    assert env.action_space.n == 3
    assert env.observation_space.low[0] == pytest.approx(-1.2)
    assert env.observation_space.high[0] == pytest.approx(0.6)
    assert env.observation_space.low[1] == pytest.approx(-0.07)
    assert env.observation_space.high[1] == pytest.approx(0.07)
    observation, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()
    assert True 
