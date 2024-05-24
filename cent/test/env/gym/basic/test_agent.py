# A simple agent showing how to interact with gym
import pytest
import gymnasium as gym

class SimpleAgent:
    def __init__(self, env):
        pass 
    
    def decide(self, observation):
        # print(observation)
        position = observation[0] 
        velocity = observation[1] 
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03, 
            0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 4 - 0.008
        # print(velocity)
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action 
    
    def learn(self, *args):
        pass 

@pytest.mark.current
def test_simple_agent():
    # MountainCar-v0
    env = gym.make("MountainCar-v0", render_mode="human")
    agent = SimpleAgent(env)
    for _ in range(1):
        episode_reward = 0
        observation, _ = env.reset()
        while True:
            action = agent.decide(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
            observation = next_observation
        print(episode_reward)
    env.close()
    assert True 
