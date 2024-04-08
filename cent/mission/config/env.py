from dataclasses import dataclass 
import os 
import json 

@dataclass 
class EnvConfig:
    env_name: str = "pc"
    dataset_root: str = "D:/dataset"    
    default_device: str = "cuda"
    log_path: str = "D:/logs/"
    result_path: str = "D:/workspace/data/result"
    doc_fig_result_path: str = "../doc/figure/result"
    doc_tab_result_path: str = "../doc/table/result"
    pretrained_path: str = "D:/pretrained/"
    blender_root: str = "D:/workspace/blender/"

    @classmethod 
    def from_json(self, json_fpath: str):
        with open(json_fpath, 'r') as f:
            config = json.load(f) 
        return self(**config)

def get_env_config(cfg_fpath: str = os.path.join(os.path.dirname(__file__),"config.json")):
    assert os.path.exists(cfg_fpath), f"Config file not found at {cfg_fpath}"
    return EnvConfig.from_json(cfg_fpath)