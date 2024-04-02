from mission.base import MissionBase
import itertools 
from app.project.nvs.gs.train import TrainGaussianProjectConfig, TrainGaussianProjectParams, TrainGaussianProject

from app.trainer.nvs.gs.basic import GaussianTrainerParams
from app.trainer.nvs.gs.vanilla import GaussianVanillaTrainerParams

class Mission(MissionBase):
    def __init__(self, config_json_file):
        super().__init__(config_json_file, __file__)
        # parse the compenent
        self.name = self.config_json["name"]
        self.dataset_name = self.config_json["dataset_name"]
        
        self.init_scene = self.config_json["init_scene"]
        
        self.train_params = self.config_json["train_params"]
        self.loss_name = self.config_json["loss_name"]
        self.trainer_name = self.config_json["trainer_name"]
        self.render_name = self.config_json["render_name"]
        self.objects = self.config_json["objects"]
        self.benchmarks = self.config_json["benchmarks"]
        self.usage = self.config_json["usage"]

        self.create_trainer_params = {
            "vanilla": GaussianVanillaTrainerParams,
            "basic": GaussianTrainerParams
        }

    def exec(self):
        proj_config = TrainGaussianProjectConfig(self.env_config)
        proj_config.name = self.name
        proj_config.usage = self.usage 
        # proj_config.sh_deg = 0
        project = TrainGaussianProject(proj_config)
        train_params = self.create_trainer_params[self.trainer_name]()

        for key in self.train_params:
            setattr(train_params, key, self.train_params[key])
        
        train_params.max_iterations = max(train_params.saving_iterations)
        for obj_name in self.objects:
            params = TrainGaussianProjectParams(
                dataset_name=self.dataset_name,
                obj_name=obj_name,
                init_scene=self.init_scene,
                trainer_name=self.trainer_name,
                render_name=self.render_name,
                train_params=train_params,
                loss_name=self.loss_name,
                metric_types=self.benchmarks
            )
            result = project.run(params)
