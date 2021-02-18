from typing import Union

import numpy as np
from scipy.stats import norm

from acquisition_functions.abstract_acquisition_function import AcquisitionFunction
from gaussian_process import GaussianProcess


class ExpectedImprovement(AcquisitionFunction):
    def _evaluate(self,
                  gaussian_process: GaussianProcess,
                  data_points: np.ndarray
                  ) -> np.ndarray:
        """
        Evaluates the acquisition function at all the data points
        :param gaussian_process:
        :param data_points: numpy array of dimension n x m where n is the number of elements to evaluate
        and m is the number of variables used to calculate the objective function
        :return: a numpy array of shape n x 1 (or a float) representing the estimation of the acquisition function at
        each point
        """
        y = gaussian_process.array_objective_function_values
        f = min(y)
        X = data_points
        n = X.shape[0]
        mean, std = GaussianProcess.get_gp_mean_std(data_points)
        EI = np.zeros(n)
        for i in range(n):
            EI[i] = (f - mean[i]) * norm.cdf(f, mean[i], std[i]) + std[i] * norm.pdf(f, mean[i], std[i])

        return EI.reshape(n, 1)


        # TODO




