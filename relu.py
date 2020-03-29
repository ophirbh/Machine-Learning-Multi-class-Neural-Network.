import numpy as np


class relu(object):
    def calc(self, mat):
        return np.maximum(mat, 0)

    def calc_gradient(self, mat):
        return 1. * (mat > 0)
        return ans
