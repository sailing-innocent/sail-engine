from ..base import ProjectConfigBase, ProjectBase 
from loguru import logger 
import os 
from typing import NamedTuple

from app.pipeline.image_classify.train import ClassifyTrainPipelineConfig, ClassifyTrainPipeline
from app.classifier.simple_mlp import SimpleMLPClassifier

# dataset
from module.dataset.classify.mnist.dataset import create_dataset as create_mnist
from module.dataset.classify.fmnist.dataset import create_dataset as create_fmnist
from module.dataset.classify.cifar10.dataset import create_dataset as create_cifar10

class ProfileClassifyProjectConfig(ProjectConfigBase):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.name = "Profile Classify Project"
        self.usage = "train_eval"
        self.output_path = ""

class ProfileClassifyProjectParams(NamedTuple):
    dataset_name: str = "fmnist"
    dataset_type: str = "image"
    loss_name: str = "cross_entropy"
    trainer_name: str = "plain" 
    batch_size: int = 32
    epochs: list = [5, 10]

class ProfileClassifyProject(ProjectBase):
    def __init__(self, config: ProfileClassifyProjectConfig):
        super().__init__(config)
        self.input_dims = {
            "mnist": 28*28,
            "fmnist": 28*28,
            "cifar10": 3*32*32,
        }
        self.create_dataset = {
            "mnist": create_mnist,
            "fmnist": create_fmnist,
            "cifar10": create_cifar10
        }

    def run(self, params: ProfileClassifyProjectParams):
        logger.info("Profile Classify Project")
        logger.info(f"Dataset: {params.dataset_name}")
        logger.info(f"Epochs: {params.epochs}")
        logger.info(f"Save at {self.target_path}")

        dataset = self.create_dataset[params.dataset_name](self.config.env_config, "train", params.batch_size)
        classes = dataset.classes()

        pipe_config = ClassifyTrainPipelineConfig(self.config.env_config)
        pipe_config.name = f"{params.dataset_name}_{params.trainer_name}_{params.loss_name}_train_pipeline"
        pipe_config.proj_name = self.config.name
        pipe_config.trainer_name = params.trainer_name
        pipe_config.epochs = params.epochs
    
        pipe = ClassifyTrainPipeline(pipe_config)
        model = SimpleMLPClassifier(
            input_dim = self.input_dims[params.dataset_name], 
            class_num = len(classes))
        process_log = pipe.run(model, dataset)

        return process_log 

