from module.metric.accuracy.classify import ClassifyAccuracy
from module.dataset.classify.base import ClassifyDataset

import torch
import torch.nn as nn

class SimpleClassifyResult:
    def __init__(self):
        self.dataset_name = ""
        self.classes = []
        self.N = 0
        self.accuracy = 0.0

    def __str__(self):
        return f"eval result on {self.dataset_name} dataset with {len(self.classes)} classes and {self.N} instances result accuracy {self.accuracy}"

class SimpleClassifyEvaluator:
    def __init__(self):
        pass 

    def eval(self, model, dataset: ClassifyDataset) -> SimpleClassifyResult:
        result = SimpleClassifyResult()
        result.dataset_name = dataset.name
        result.classes = dataset.classes()
        N = len(dataset)
        result.N = N 

        targets = []
        predicts = []
        with torch.no_grad():
            for i in range(N):
                sample, target = dataset[i]
                targets.append(target)
                distribution = model(sample.unsqueeze(0))
                distribution = nn.functional.softmax(distribution, dim=-1)[0]     
                predict = distribution.argmax()
                predicts.append(predict)
           
        metric = ClassifyAccuracy(targets, predicts, N)
        result.accuracy = metric.value 
        return result 


