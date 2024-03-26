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
        self.objects = self.config_json["objects"]
        self.benchmarks = self.config_json["benchmarks"]
        self.usage = self.config_json["usage"]

        self.train_params_list = self.config_json["train_params_list"]
        self.loss_names = self.config_json["loss_names"]
        self.render_names = self.config_json["render_names"]
        self.trainer_names = self.config_json["trainer_names"]

  
        self.create_train_params = {
            "vanilla": GaussianVanillaTrainerParams,
            "basic": GaussianTrainerParams
        }

    def exec(self):
        proj_config = TrainGaussianProjectConfig(self.env_config)
        proj_config.name = self.name
        proj_config.usage = self.usage 
        # proj_config.sh_deg = 0
        project = TrainGaussianProject(proj_config)
        

        for render_name, trainer_name, _train_params, loss_name in itertools.product(self.render_names, self.trainer_names, self.train_params_list, self.loss_names):
            print(f"render_name: {render_name}, trainer_name: {trainer_name}, train_params: {_train_params}, loss_name: {loss_name}")
            train_params = self.create_train_params[trainer_name]()
            for key in _train_params:
                setattr(train_params, key, _train_params[key])
            
            train_params.max_iterations = max(train_params.saving_iterations)
            for obj_name in self.objects:
                params = TrainGaussianProjectParams(
                    dataset_name=self.dataset_name,
                    obj_name=obj_name,
                    trainer_name=trainer_name,
                    render_name=render_name,
                    train_params=train_params,
                    loss_name=loss_name,
                    metric_types=self.benchmarks
                )
                project.run(params)
