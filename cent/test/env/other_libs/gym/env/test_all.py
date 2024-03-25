import pytest 
import gymnasium as gym

@pytest.mark.current 
def test_all():
    env_specs = gym.envs.registry
    print([env_spec for env_spec in env_specs])
    assert True