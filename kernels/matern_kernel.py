import numpy as np

from kernels.abstract_kernel import Kernel


class MaternKernel(Kernel):
    def get_covariance_matrix(self, X: np.ndarray, Y: np.ndarray):
        """
        :param X: numpy array of size n_1 x m for which each row (x_i) is a data point at which the objective function can be evaluated
        :param Y: numpy array of size n_2 x m for which each row (y_j) is a data point at which the objective function can be evaluated
        :return: numpy array of size n_1 x n_2 for which the value at position (i, j) corresponds to the value of
        k(x_i, y_j), where k represents the kernel used.
        """
        sigma_f_square = np.exp(self.log_amplitude) ** 2
        l = np.exp(self.log_length_scale)
        n_1 = X.shape[0]
        n_2 = X.shape[0]
        K = np.zeros((n_1, n_2))
        for i in range(n_1):
            for j in range(n_2):
                res = np.sqrt(3) * np.sqrt(np.sum((X[i] - Y[j]) ** 2)) / l
                K[i, j] = sigma_f_square * (1 + res) * np.exp(-res)

        return K

        # TODO
