import numpy as np
import scipy.linalg


class rls(object):
    """docstring for ClassName"""

    def __init__(self, lbd, theta, nn_dim, output_dim):

        self.lbd = lbd
        self.nn_dim = nn_dim
        self.y_dim = output_dim
        self.draw = []

        self.theta = theta * np.ones([self.nn_dim, self.y_dim])

        self.initialize()

    def initialize(self):
        self.rls_state = []

        self.F = 1000 * np.eye(self.nn_dim)

        self.F_M = self.F

        for i in range(self.y_dim - 1):
            self.F_M = scipy.linalg.block_diag(self.F_M, self.F)

    # rls.update(hidden_vec, obs_Y[i,:])
    def update(self, hidden_vec, obs_Y):
        hidden_vec = np.concatenate([hidden_vec, np.ones([1, 1])], axis=1)
        # print("original pred:")
        # print(hidden_vec @ self.theta)
        # print("ground truth:")
        # print(obs_Y)
        for j in range(self.y_dim):
            self.F = self.F_M[self.nn_dim * j:self.nn_dim * (j + 1),
                     self.nn_dim * j:self.nn_dim * (j + 1)]
            # print(self.F)
            k = self.lbd + hidden_vec @ self.F @ hidden_vec.T
            # 65 * 1
            k = self.F @ hidden_vec.T / k
            # print("rls debug info:")
            # print(k, hidden_vec @ k)
            # print("original theta:")
            # print(self.theta)
            # print("theta has updated:")
            # print(k @ (obs_Y[:, j] - hidden_vec @ self.theta[:, j]))
            self.theta[:, j] = self.theta[:, j] + k @ (obs_Y[:, j] - hidden_vec @ self.theta[:, j])
            # print("updated theta:")
            # print(self.theta)

            # self.F = (self.F - k @ hidden_vec @ self.F) / self.lbd

            self.F_M[self.nn_dim * j:self.nn_dim * (j + 1),
            self.nn_dim * j:self.nn_dim * (j + 1)] = self.F

        pred = hidden_vec @ self.theta
        # print("updated pred:")
        # print(pred)

        self.rls_state.append(pred)

    def predict(self, hidden_vec):
        hidden_vec = np.concatenate([hidden_vec, np.ones([1, 1])], axis=1)
        # print("Roll out action:")
        # print(hidden_vec, self.theta)
        # print(hidden_vec @ self.theta)
        return hidden_vec @ self.theta