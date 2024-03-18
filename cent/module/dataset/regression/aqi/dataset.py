from module.config.env import BaseEnvConfig
from ..base import RegressionDatasetConfig, RegressionDataset, data_iter
import csv
import os 
import numpy as np 
import torch 

class AQIDatasetConfig(RegressionDatasetConfig):
    def __init__(self, env_config: BaseEnvConfig):
        super().__init__(env_config)
        self.input_dim = 10
        self.output_dim = 1
        self.batch_size = 10

    def dataset_root(self):
        return self.env_config.dataset_root() 

    @property 
    def csv_file_path(self):
        return os.path.join(self.dataset_root(), "aqi", "dataset.csv")

class AQIDataset(RegressionDataset):
    def __init__(self, config: AQIDatasetConfig):
        super().__init__(config)
        self.name = "aqi dataset"
        self._data = []
        # read the data from dataset, the data looks like
        ###################################################
        ## One Raw starts with row 1
        ## 0  ## City Name             ## Ngawa Prefecture
        ## 1  ## AQI                   ## 23
        ## 2  ## Precipitation         ## 665.1
        ## 3  ## GDP                   ## 271.13
        ## 4  ## Temperature           ## 8.2 
        ## 5  ## Longitude             ## 102.22465
        ## 6  ## Latitude              ## 31.89941
        ## 7  ## Altitude              ## 2617
        ## 8  ## Population Density    ## 11
        ## 9  ## Coastal               ## 0
        ## 10 ## Green CoverageRate    ## 36
        ## 11 ## Incineration(10000ton)## 23
        ####################################################
        with open(config.csv_file_path, "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0: # first row is name, pass
                    continue
                self._data.append([float(item) for item in row[1:]])
            f.close()

        self._data = np.array(self._data)
        N = len(self._data)
        self._data = torch.from_numpy(self._data).float()
        assert N == 323

    def features(self):
        return self._data[:, 1:]

    def labels(self):
        return self._data[:, 0]

    def __iter__(self):
        return data_iter(self.config.batch_size, self.features(), self.labels())

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        return self._data[index]
