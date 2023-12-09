import os 
from loguru import logger 

class ProjectConfigBase:
    def __init__(self, env_config):
        self.env_config = env_config 
        self.name = "dummy_project"

class ProjectBase:
    def __init__(self, config: ProjectConfigBase):
        self.config = config 
        self.target_path = os.path.join(config.env_config.log_path, config.name)
        os.makedirs(self.target_path, exist_ok=True)
        logger.info(f"init project {self.config.name} in {self.target_path}")
