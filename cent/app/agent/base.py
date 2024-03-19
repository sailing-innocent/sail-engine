from abc import ABC, abstractmethod

class AgentConfigBase:
    def __init__(self, env_config):
        self.env_config = env_config 

class AgentBase(ABC):
    def __init__(self, config: AgentConfigBase):
        pass 

    @abstractmethod
    def reset(self):
        pass 

    @abstractmethod
    def observe(self, world):
        pass 

    @abstractmethod
    def act(self):
        pass

    @abstractmethod 
    def collect_reward(self, reward):
        pass

    @abstractmethod
    def update(self):
        pass

    # load 
    # save