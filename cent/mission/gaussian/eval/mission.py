import os 
from mission.base import BaseMission 
from app.project.nvs.gs.eval import EvalGaussianProjectConfig, EvalGaussianProjectParams, EvalGaussianProject

def get_ply_from_json(gs_ply_json, parent_path = ""):
    return os.path.join(parent_path, gs_ply_json['path'], "_".join([gs_ply_json['dataset'], gs_ply_json['obj_name'], str(gs_ply_json["iter"])]) + ".ply")

class Mission(BaseMission):
    def __init__(self, config_json_file):
        super().__init__(config_json_file, __file__)
        self.name = self.config_json["name"]
        self.render_name = self.config_json["render_name"]
        self.usage = self.config_json["usage"]
        self.checkpoints = self.config_json["checkpoints"]

    def exec(self):
        proj_config = EvalGaussianProjectConfig(self.env_config)
        proj_config.name = self.name
        proj_config.usage = self.usage
        project = EvalGaussianProject(proj_config)
    
        for ckpt_json in self.checkpoints:
            ply_file_path = get_ply_from_json(ckpt_json, self.env_config.pretrained_path)
            params = EvalGaussianProjectParams(
                dataset_name = ckpt_json["dataset"],
                obj_name = ckpt_json["obj_name"],
                ckpt_path = ply_file_path,
                output_name =  ckpt_json["dataset"] + "_" + ckpt_json["obj_name"],
                render_name =  self.render_name)
            project.run(params)