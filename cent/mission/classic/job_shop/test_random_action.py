
import pytest 
from cent.app.scene.gym_env.jss.jss_env import JssEnv 
import numpy as np 
from pathlib import Path

@pytest.mark.current 
def test_job_shop_env():
    env_config = {
        "instance_path": Path(__file__).parent.absolute() / "instances" / "ta80"
    }
    env = JssEnv(env_config)
    obs = env.reset()
    done = False
    cum_reward = 0
    while not done:
        legal_actions = obs["action_mask"]
        actions = np.random.choice(
            len(legal_actions), 1, p=(legal_actions / legal_actions.sum())
        )[0]
        print(actions)
        obs, rewards, done, _ = env.step(actions)
        cum_reward += rewards
    print(f"Cumulative reward: {cum_reward}")