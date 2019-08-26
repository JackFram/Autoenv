import envs.hyperparams as hyperparams
import os
import numpy as np
from envs import build_env


if __name__ == '__main__':
    params = hyperparams.parse_args()
    np.savez("./args/params.npz", args=params)

