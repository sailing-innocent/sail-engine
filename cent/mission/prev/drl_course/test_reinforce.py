import pytest 
import numpy as np 
import torch 

from module.config.env import get_env_config_by_name
from app.trainer.rl.policy_gradient.trainer import PolicyGradientTrainer, PolicyGradientTrainerConfig, PolicyGradientTrainProcessLog

@pytest.mark.current 
def test_reinforce_inverted_pendulum():
    env_config = get_env_config_by_name("pc")
    trainer_config = PolicyGradientTrainerConfig(env_config)
    trainer = PolicyGradientTrainer(trainer_config)
    trainer.train()
    