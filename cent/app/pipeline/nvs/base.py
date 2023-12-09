from ..base import PipelineConfigBase, PipelineBase

class NVSPipelineConfig(PipelineConfigBase):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.name = "nvs_pipeline"
        self.dataset_name = "nerf_blender"
        self.obj_name = "lego"

    def dataset_root(self):
        return self.env_config.dataset_root()

class NVSPipeline(PipelineBase):
    def __init__(self, config: NVSPipelineConfig):
        super().__init__(config)

    def run(self):
        super().run()
