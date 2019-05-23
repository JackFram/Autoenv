# Zhihao Zhang
# MvNormal class in python

import math
import numpy as np


class MvNormal:
    def __init__(self, mu: np.array, cov: np.array):
        self.mu = mu
        self.cov = cov

    def rand_sample(self):
        return np.random.multivariate_normal(self.mu, self.cov)



