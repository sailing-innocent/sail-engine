import os 
from mission.base import MissionBase 
from app.project.nvs.gs.eval import EvalGaussianProjectConfig, EvalGaussianProjectParams, EvalGaussianProject

def get_ply_from_json(gs_ply_json, parent_path = ""):
    return os.path.join(parent_path, gs_ply_json['ckpt_path'], "_".join([gs_ply_json['dataset_name'], gs_ply_json['obj_name'], str(gs_ply_json["iter"])]) + ".ply")

class Mission(MissionBase):
    def __init__(self, config_json_file):
        super().__init__(config_json_file, __file__)
        self.name = self.config_json["name"]
        self.render_name = self.config_json["render_name"]
        self.usage = self.config_json["usage"]
        # self.checkpoints = self.config_json["checkpoints"]
        self.scenes = self.config_json["scenes"]

    def exec(self):
        proj_config = EvalGaussianProjectConfig(self.env_config)
        proj_config.name = self.name
        proj_config.usage = self.usage
        project = EvalGaussianProject(proj_config)
    
        for scene in self.scenes:
            if scene["type"] == "ckpt":
                ply_file_path = get_ply_from_json(scene, self.env_config.pretrained_path)
                scene["ckpt_path"] = ply_file_path
                params = EvalGaussianProjectParams(
                    scene = scene,
                    output_name =  scene["dataset_name"] + "_" + scene["obj_name"],
                    render_name =  self.render_name)
            elif scene["type"] == "sphere":
                params = EvalGaussianProjectParams(
                    output_name =  "_".join(["sphere", str(scene["r"]), str(scene["N"])]), 
                    render_name =  self.render_name,
                    scene = scene)
            project.run(params)