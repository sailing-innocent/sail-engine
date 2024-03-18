from abc import ABC, abstractmethod 

class VisualizerConfigBase(ABC):
    def __init__(self, env_config):
        self.env_config = env_config
    
class VisualizerBase(ABC):
    def __init__(self, config: VisualizerConfigBase):
        self.config = config 

    @abstractmethod
    def visualize(self, dataloader):
        pass
