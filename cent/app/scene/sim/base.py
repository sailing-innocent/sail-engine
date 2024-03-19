# Scene for Simulation
from abc import ABC, abstractmethod

class SimSceneConfig:
    def __init__(self, env_config):
        self.env_config = env_config

class SimSceenBase(ABC):
    def __init__(self, config: SimSceneConfig):
        self.config = config

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self):
        # scene changes
        pass