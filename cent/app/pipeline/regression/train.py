from ..base import PipelineConfigBase, PipelineBase

from module.dataset.regression.base import RegressionDataset, RegressionDatasetConfig
from module.dataset.regression.linear.dataset import LinearDataset, LinearDatasetConfig
from module.dataset.regression.xsinx.dataset import XSNIXDataset1D, XSNIXDataset1DConfig
from module.dataset.regression.aqi.dataset import AQIDataset, AQIDatasetConfig
# model
from module.model.linear.simple import linreg

# evaluator
from app.evaluator.regression import RegressionResult, RegressionEvaluator
# trainer
from app.trainer.regression.base import RegressionTrainerConfig, RegressionTrainerProcessLog, RegressionTrainer

import os 
import json
import torch

import matplotlib.pyplot as plt

class RegressionTrainEvalPipelineConfig(PipelineConfigBase):
    def __init__(self, env_config):
        super().__init__(env_config)
        self.exp_name = "regression"
        self.dataset_name = "linear"
        self.model_name = "linreg"
        self.report_show = False
        self.report_save = True 
        self.epochs_list = [1, 3, 5]
        self.batch_size_list = [10, 30, 50]

class RegressionTrainEvalParams:
    def __init__(self):
        self.epoch = 3
        self.batch_size = 10

class RegressionTrainEvalPipeline(PipelineBase):
    def __init__(self, config: RegressionTrainEvalPipelineConfig):
        """
        Inherited
            eval_results: list
            train_process_logs: list
            weights: list
            config: MLPipelineConfig
        """
        super().__init__(config)
        self.done = False
        self.N_epochs = len(config.epochs_list)
        self.N_batch_size = len(config.batch_size_list) 
        self.N_total = self.N_epochs * self.N_batch_size
        self.set_dataset_config()
        self.model = linreg 
        self.loss = torch.functional.mse_loss
        self.evaluator = RegressionEvaluator()

    def set_dataset_config(self):
        if self.config.dataset_name == "linear":
            self.dataset_config = LinearDatasetConfig(self.config.env_config)
        elif self.config.dataset_name == 'xsinx':
            self.dataset_config = XSNIXDataset1DConfig(self.config.env_config)
        elif self.config.dataset_name == "aqi":
            self.dataset_config = AQIDatasetConfig(self.config.env_config)
        else:
            raise NotImplementedError('dataset {} is not implemented', config.dataset_name)
    
    def get_dataset_by_config(self, config: RegressionDatasetConfig):
        if self.config.dataset_name == "linear":
            return LinearDataset(config)
        elif self.config.dataset_name == 'xsinx':
            return XSNIXDataset1D(config)
        elif self.config.dataset_name == "aqi":
            return AQIDataset(config)
        else:
            raise NotImplementedError('dataset {} is not implemented', config.dataset_name)

    def run(self):
        super().run() # get id, init 
        # self.target_path, self.meta_file_path
        for i in range(self.N_total):
            idx = i
            epochs_idx = i % self.N_epochs
            epochs = self.config.epochs_list[epochs_idx]
            i = i // self.N_epochs
            batch_size_idx = i % self.N_batch_size
            i = i // self.N_batch_size

            batch_size = self.config.batch_size_list[batch_size_idx]
            self.logger.info("running pipeline with epochs {}, batch size {}".format(
                    epochs, batch_size))

            self.dataset_config.batch_size = batch_size
            self.train_dataset = self.get_dataset_by_config(self.dataset_config)
            self.eval_dataset = self.get_dataset_by_config(self.dataset_config)
            
            self.run_one_param(epochs, idx)

    def run_one_param(self, epochs: int, i: int):
        trainer_config = RegressionTrainerConfig(self.config.env_config)
        trainer_config.epochs = epochs
        trainer = RegressionTrainer(trainer_config)

        # Init
        w = torch.normal(0, 0.01, size=(
            self.dataset_config.input_dim,
            self.dataset_config.output_dim), requires_grad=True)
        b = torch.zeros(self.dataset_config.output_dim, requires_grad=True)
        # Eval Init
        eval_result = self.evaluator.eval(self.eval_dataset, self.model, (w, b))
        self.logger.info(f'init eval loss {eval_result.average_error:f}')
        # Train
        train_process_log = trainer.train(self.train_dataset, self.model, self.loss, (w, b))
        if self.config.save_train:
            log_file_name = "train_process_log_"+str(i)
            log_file_path = os.path.join(self.target_path, log_file_name +".json")
            train_process_log.save(log_file_path)
            self.train_process_logs.append(log_file_name)

        # Eval
        eval_result = self.evaluator.eval(self.eval_dataset, self.model, (w, b))
        self.logger.info(f'final eval loss {eval_result.average_error:f}')

        if self.config.save_eval:
            eval_result_file_name = "eval_result_"+str(i)
            eval_result_file_path = os.path.join(self.target_path, eval_result_file_name + ".json")
            eval_result.save(eval_result_file_path)
            self.eval_results.append(eval_result_file_name)
        
    def report(self):
        self.logger.info("reporting the pipeline from {}".format(self.target_path))
        self.logger.info("reading meta from {}".format(self.meta_file_path))
        with open(self.meta_file_path, "r") as f:
            meta = json.load(f)

        train_log_names = meta["train_process_logs"]
        for train_log_name in train_log_names:
            train_log_path = os.path.join(self.target_path, train_log_name + ".json")
            with open(train_log_path, "r") as f:
                train_log = json.load(f)
            
            batch_loss = train_log["batch_loss"]
            plt.plot(
                [item["batch"] for item in batch_loss], 
                [item["loss"] for item in batch_loss])

        # generate loss figure
        plt.xlabel("batched step")
        plt.ylabel("loss")
        plt.title("Train Loss")
        plt.legend([
            'Epochs {} Batch Size {}'.format(
                epoch, batch_size)
            for batch_size in self.config.batch_size_list for epoch in self.config.epochs_list])
        if self.config.report_show:
            plt.show()
        if self.config.report_save:
            report_file_path = os.path.join(self.target_path, "report.png")
            plt.savefig(report_file_path)

        eval_result_names = meta["eval_results"]
        eval_results = []
        for eval_result_name in eval_result_names:
            eval_result_path = os.path.join(self.target_path, eval_result_name + ".json")
            with open(eval_result_path, "r") as f:
                eval_result = json.load(f)
                eval_results.append(eval_result)

        # generate eval table
        table_json = {
            "caption": "Eval Results",
            "label": "tab:eval_results",
            "cols": self.config.batch_size_list,
            "rows": self.config.epochs_list,
            "data": []
        }
        table_data = []
        for i, epoch in enumerate(self.config.epochs_list):
            row_data = []
            for j, batch_size in enumerate(self.config.batch_size_list):
                idx = i + j * self.N_epochs
                eval_result = eval_results[idx]
                row_data.append("{:.4f}".format(eval_result["average_error"]))
            table_data.append(row_data)
        table_json["data"] = table_data
        table_file_path = "doc/table/result/tab_regression.json"
        with open(table_file_path, "w") as f:
            json.dump(table_json, f)