import os 
from mission.base import MissionBase
from app.project.image_classify.profile import ProfileClassifyProjectConfig, ProfileClassifyProjectParams, ProfileClassifyProject
from module.utils.tex.figure import TexFigure, TexFigData

class Mission(MissionBase):
    def __init__(self, config_json_file):
        super().__init__(config_json_file, __file__)
        # parse the compenent
        self.name = self.config_json["name"]
        self.dataset_name = self.config_json["dataset_name"]
        self.epochs = self.config_json["epochs"]
        self.loss_name = self.config_json["loss_name"]
        self.trainer_name = self.config_json["trainer_name"]
        self.usage = self.config_json["usage"]
        self.result_json = self.config_json["result"]
        self.batch_size_list = self.config_json["batch_size_list"]

    def run_proj(self):
        proj_config = ProfileClassifyProjectConfig(self.env_config)
        proj_config.name = self.name
        project = ProfileClassifyProject(proj_config)
        for batch_size in self.batch_size_list:
            params = ProfileClassifyProjectParams(
                dataset_name=self.dataset_name,
                trainer_name=self.trainer_name,
                epochs=self.epochs,
                batch_size=batch_size,
                loss_name=self.loss_name
            )
            process_log = project.run(params)
            yield [[pair["item"], pair["loss"]] for pair in process_log.item_loss_pairs]

    def exec(self):
        results = self.run_proj()
        fig = TexFigure()
        fig.title = self.result_json["title"]
        output_path = os.path.join(self.env_config.doc_fig_result_path, self.result_json["output_path"])
        os.makedirs(output_path, exist_ok=True)
        for idx, result in enumerate(results):
            dt = TexFigData()
            dt.data = result 
            fig.add_line(dt, "simple mlp batchsize " + str(self.batch_size_list[idx]))
        fig.export(output_path)
