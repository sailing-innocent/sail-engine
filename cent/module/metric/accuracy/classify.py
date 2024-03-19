"""
classify accuracy metric
"""

import numpy as np

class ClassifyAccuracy:
    def __init__(self, target, predicted, N):
        self.hit = 0
        self.N = N
        for (t, p) in zip(target, predicted):
            if t == p:
                self.hit += 1

    @property
    def value(self):
        return self.hit / self.N

    def __str__(self):
        return "accuracy: {}".format(self.value)