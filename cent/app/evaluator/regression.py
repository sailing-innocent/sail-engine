# Regression Evaluator
# - dataset name
# - dataset description
# - nums of samples
# - average_error
# - max_error
import json 
import torch 

class RegressionResult:
    def __init__(self):
        self.dataset_name = ""
        self.N = 0
        self.average_error = 0.0
        self.max_error = 0.0

    def save(self, target_path: str):
        with open(target_path, "w") as f:
            json.dump(self.__dict__, f)

class RegressionEvaluator:
    def __init__(self):
        self.metric_type = "errors"
        self.metric = torch.functional.mse_loss
        super().__init__()

    def eval(self, dataset, model, params):
        result = RegressionResult()
        max_error = 0
        average_error = 0
        result.dataset_name = str(dataset)
        result.N = len(dataset)
        w, b = params 
        for idx in range(result.N):
            X, y = dataset[idx]
            y_hat = model(X, w, b)
            error = self.metric(y, y_hat)
            average_error += error
            if error > max_error:
                max_error = error
        average_error = average_error / result.N
        result.average_error = average_error.item()
        result.max_error = max_error.item()
        return result


    