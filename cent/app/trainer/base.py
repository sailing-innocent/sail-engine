# Base Network Trainer

from abc import ABC, abstractmethod

class TrainerConfigBase(ABC):
    """TrainerConfigBase(ABC):
    - Define the Train and Eval Dataset
    - Define the Metrics
    - Define the Optimizer
    - Define the Loss Function
    """
    def __init__(self, env_config):
        self.env_config = env_config

class TrainProcessLogBase(ABC):
    """TrainProcessLog(ABC):
    - The Train Process Log is the abstraction of one training process returned by the train() method of TrainerBase.
    - One can also build external Train Log Via .pth file or other methods
    """
    def __init__(self):
        pass 

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass 

class TrainerBase(ABC):
    """TrainerBase(ABC):
    - config: TrainerConfigBase
    - logs: List[TrainProcessLogBase]
    - train(): -> log
    - eval(log: TrainProcessLogBase) -> Metrics
    """

    def __init__(self, config: TrainerConfigBase):
        self.config = config

    @abstractmethod 
    def train(self) -> TrainProcessLogBase:
        pass 
