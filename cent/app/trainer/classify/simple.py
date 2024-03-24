from ..base import TrainerConfigBase, TrainProcessLogBase, TrainerBase 

import torch 
import torch.nn as nn 
import os 
from loguru import logger 

class SimpleClassifyTrainerConfig(TrainerConfigBase):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.optimizer_name: str = "sgd"
        self.lr: float = 0.001
        self.epochs = [5, 10]
        self.log_batch_interval: int = 1000
        self.save_path: str = ""

class SimpleClassifyTrainProcessLog(TrainProcessLogBase):
    def __init__(self):
        super().__init__()
        self.item_loss_pairs = []
        self.optimizer_name = ""
        self.dataset_name = ""
        self.model_name = ""
        self.batch_size = 1

    def save(self, target_path: str):
        log = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "optimizer": self.optimizer_name,
            "batch_size": self.batch_size,
            "item_loss_pair": self.item_loss_pair,
        }
        log_file_path = os.path.join(target_path, "train_process_log.json")
        with open(log_file_path, "w") as f:
            json.dump(log, f)

    def load(self, file_path: str):
        # for eval, no need to load batch_loss
        with open(file_path, "r") as f:
            log = json.load(f)
        self.dataset_name = log["dataset"]
        self.model_name = log["model"]
        self.optimizer_name = log["optimizer"]

class SimpleClassifyTrainer(TrainerBase):
    def __init__(self, config: SimpleClassifyTrainerConfig):
        super().__init__(config)
        self.name = "classify_trainer"
        optimizers = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam
        }
        self.optimizer_factory = optimizers[config.optimizer_name]

    def train(self, model, dataset, loss_fn):
        # train the model on such dataset 
        optimizer = self.optimizer_factory(model.parameters(), lr=self.config.lr)
        process_log = SimpleClassifyTrainProcessLog()
        process_log.optimizer_name = self.config.optimizer_name
        process_log.dataset_name = dataset.name
        process_log.model_name = model.name
        process_log.batch_size = dataset.batch_size

        i = 0
        for epoch in range(self.config.epochs[-1]):
            logger.info("Epoch: {}".format(epoch))
            for batch_idx, (sample, label) in enumerate(dataset):
                i = i + 1
                # forward & backward
                predict = model(sample)
                loss = loss_fn(predict, label)
                loss.backward()

                with torch.no_grad():
                    # optimizer step
                    optimizer.step()
                    optimizer.zero_grad()

                    # log
                    if i % self.config.log_batch_interval == 0:
                        logger.info(f"batch {batch_idx}, loss {loss.item()}")
                        process_log.item_loss_pairs.append({
                            "item": i * dataset.batch_size,
                            "loss": loss.item()
                        })
            # save
            with torch.no_grad():
                if (epoch + 1) in self.config.epochs:
                    self.save_dict(model, epoch)
        
        return process_log 

    def save_dict(self, model, i: int):
        model_path = os.path.join(self.config.save_path, "model_"+str(i)+".pth")
        os.makedirs(self.config.save_path, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logger.info("Saved PyTorch Model State to {}".format(model_path))