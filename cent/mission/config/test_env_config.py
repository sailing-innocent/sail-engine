import pytest 

from mission.config.env import get_env_config

@pytest.mark.current 
def test_env_config():
    config = get_env_config()
    assert config.env_name == "pc"
    assert True 