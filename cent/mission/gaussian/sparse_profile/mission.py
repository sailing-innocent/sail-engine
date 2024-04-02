from mission.base import MissionBase 
import itertools 
from app.project.nvs.sparse_gs.train import TrainGaussianProjectConfig, TrainGaussianProjectParams, TrainGaussianProject
from app.trainer.nvs.sparse_gs.basic import GaussianTrainerParams
from app.trainer.nvs.sparse_gs.vanilla import GaussianVanillaTrainerParams
from app.trainer.nvs.sparse_gs.epipolar import  GaussianEpipolarTrainerParams


from module.utils.tex.table import TexTable 
import os 

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
        self.result_template = self.config_json["result_template"]
        self.init_scenes = self.config_json["init_scenes"]

        self.create_train_params = {
            "vanilla": GaussianVanillaTrainerParams,
            "basic": GaussianTrainerParams,
            "epipolar": GaussianEpipolarTrainerParams
        }

    def exec(self):
        proj_config = TrainGaussianProjectConfig(self.env_config)
        proj_config.name = self.name
        proj_config.usage = self.usage 
        # proj_config.sh_deg = 0
        self.project = TrainGaussianProject(proj_config)
        results = self.run_project()
        
    def run_through_objects(self, trainer_name, render_name, train_params, loss_name, init_scene):
        for obj_name in self.objects:
            init_scene["obj_name"] = obj_name
            params = TrainGaussianProjectParams(
                dataset_name=self.dataset_name,
                obj_name=obj_name,
                init_scene = init_scene,
                trainer_name=trainer_name,
                render_name=render_name,
                train_params=train_params,
                loss_name=loss_name,
                metric_types=[b["name"] for b in self.benchmarks]
            )
            result = self.project.run(params)
            result["name"] = obj_name
            yield result

    def run_project(self):
        n_objs = len(self.objects)
        full_result_tabs = {}
        suffix = ".json"
        result_path = self.result_template["path"]

        for benchmark_json in self.benchmarks:
            benchmark = benchmark_json["name"]
            full_result_tab = TexTable(0, n_objs + 1)
            full_result_tab.caption = f"3DGS Profile Results ({benchmark})"
            full_result_tabs[benchmark] = full_result_tab
            # add to template 
            if benchmark_json["use_template"]:
                template = self.result_template["template"]
                template_file_name = os.path.join(result_path, template.replace("{benchmark}", benchmark) + suffix)
                template_tab = TexTable(1, n_objs + 1)
                template_tab.from_json_file(template_file_name)
                full_result_tabs[benchmark].append_rows(template_tab)
        
        for render_name, trainer_name, _train_params, loss_name, init_scene_json in itertools.product(
            self.render_names, self.trainer_names, self.train_params_list, self.loss_names, self.init_scenes):
            init_scene = {
                "type": init_scene_json["type"],
                "ckpt_path": init_scene_json["ckpt_path"],
                "dataset_name": self.dataset_name,
                "obj_name": "",
                "postfix": init_scene_json["postfix"]
            }
            print(f"render_name: {render_name}, trainer_name: {trainer_name}, train_params: {_train_params}, loss_name: {loss_name}")
            train_params = self.create_train_params[trainer_name]()
            for key in _train_params:
                setattr(train_params, key, _train_params[key])
            
            train_params.max_iterations = max(train_params.saving_iterations)
            # THROUGH OBJECTS
            results = self.run_through_objects(trainer_name, render_name, train_params, loss_name, init_scene)


            # PARSE RESULT
            for benchmark in full_result_tabs.keys():
                i = 0
                average = 0.0
                result_tab = TexTable(1, n_objs + 1)

                # collect the name
                result_tab.rows[0] = "3dgs " + "_".join([
                    render_name if len(self.render_names) > 1 else "",
                    trainer_name if len(self.trainer_names) > 1 else "",
                    loss_name if len(self.loss_names) > 1 else "",
                    train_params.name if len(self.train_params_list) > 1 else "",
                    init_scene["postfix"] if len(self.init_scenes) > 1 else ""])
            
                for result in results:
                    result_data = result[benchmark]
                    average += result_data 
                    result_tab[0, i] = "{:.2f}".format(float(result_data))
                    result_tab.cols[i] = result["name"]
                    i = i + 1
                
                # average
                average = average / n_objs
                result_tab[0, i] = "{:.2f}".format(float(average))
                result_tab.cols[i] = "average"
                full_result_tabs[benchmark].append_rows(result_tab)
            
        # EXPORT
        for benchmark in full_result_tabs.keys():    
            result_name = self.result_template["name"].replace("{benchmark}",benchmark)
            result_name = result_name + f"_{self.dataset_name}"
            target_file_path = os.path.join(result_path, result_name + suffix)
            full_result_tabs[benchmark].to_json_file(target_file_path)

