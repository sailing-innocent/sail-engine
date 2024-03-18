import os 
import json 
from loguru import logger 

class PipelineConfigBase:
    def __init__(self, env_config):
        self.env_config = env_config
        self.name = "dummy_pipeline"
        self.proj_name = "dummy_project"

class PipelineBase:
    def __init__(self, config: PipelineConfigBase):
        self.config = config
        # if name not set, use id instead
        name = config.name
        self.target_path = os.path.join(config.env_config.log_path, config.proj_name, name)
        os.makedirs(self.target_path, exist_ok=True)

    def run(self):
        meta_file_name = "pipeline.json"
        self.meta_file_path = os.path.join(self.target_path, meta_file_name)
        logger.info("Saving pipeline meta file to {}".format(self.meta_file_path))
        with open(self.meta_file_path, "w") as f:
            meta = self.config.__dict__.copy() # fuck 好坑
            meta["env_config"] = ""
            json.dump(meta, f, indent=4)
