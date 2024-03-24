from ..base import PipelineBase, PipelineConfigBase
# trainer
from app.trainer.classify.simple import SimpleClassifyTrainerConfig, SimpleClassifyTrainer
from loguru import logger 
import torch 

class ClassifyTrainPipelineConfig(PipelineConfigBase):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.name = "Classify Train Pipeline"
        # componenet
        self.trainer_name = ""
        self.evaluator_name = ""
        self.epochs = [5, 10]
        # configure
        self.save_eval_result = False 
        self.save_process_log = False 

class ClassifyTrainPipeline(PipelineBase):
    def __init__(self, config: ClassifyTrainPipelineConfig):
        super().__init__(config)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        trainer_config = SimpleClassifyTrainerConfig(config.env_config)
        trainer_config.save_path = self.target_path
        trainer_config.epochs = config.epochs
        self.trainer = SimpleClassifyTrainer(trainer_config)

    def run(self, model, dataset):
        logger.info(f"train with {self.trainer.name}")
        process_log = self.trainer.train(model, dataset, self.loss_fn)
        return process_log 
        