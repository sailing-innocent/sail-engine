from abc import ABC, abstractmethod
import os 
import json 
from mission.config.env import get_env_config 

# Fuck Matplotlib and PIL
import logging 
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.INFO)
pil_logger = logging.getLogger('PIL')  
pil_logger.setLevel(logging.INFO)

class BaseMission(ABC):
    def __init__(self, config_json_file, current_f=__file__):
        with open(os.path.join(os.path.dirname(current_f), config_json_file), 'r') as config_json_f:
            self.config_json = json.loads(config_json_f.read())
            config_json_f.close()
        self.env_config = get_env_config()

    @abstractmethod
    def exec(self):
        pass