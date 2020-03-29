import numpy as np


class smart_min_max(object):

    def __init__(self, vec_lenth):
        self.vec_lenth = vec_lenth
        self.min_array = np.zeros(vec_lenth)
        self.max_array = np.zeros(vec_lenth)
        self.num_of_activation = 0

    def normalize(self, mat, new_min_arr, new_max_arr):
        mat = mat.astype(float)

        minarr = np.amin(mat, 0)
        maxarr = np.amax(mat, 0)

        # If one of the parameters contains only one value avoid zero range
        for i in range(0, len(mat[0])):
            if minarr[i] == maxarr[i]:
                mat[0, i] = minarr[i] + 1

        # Recalculate the max value after the change was made to the data
        # in order to avoid zero range
        original_max = maxarr
        maxarr = np.amax(mat, 0)

        old_range = np.subtract(maxarr, minarr)
        new_range = np.subtract(new_max_arr, new_min_arr)
        rang = np.divide(new_range, old_range)

        # Do the normalization on the matrix's lines
        for i in range(0, len(mat)):
            mat[i] = np.subtract(mat[i], np.transpose(minarr))
            mat[i] = np.multiply(mat[i], np.transpose(rang))
            mat[i] = np.add(mat[i], np.transpose(new_min_arr))

        # If one of the parameters contains only one value fix the value was change to avoid zero range
        for i in range(0, len(mat[0])):
            if minarr[i] == original_max[i]:
                mat[:, i] = new_max_arr[i]

        # Update normaliztion history
        self.min_array = np.add(self.min_array, minarr)
        self.max_array = np.add(self.max_array, maxarr)
        self.num_of_activation = self.num_of_activation + 1

        return mat

    def smart_normalize(self, data, new_min_arr, new_max_arr):
        if self.num_of_activation == 0:
            return data

        # Get the averegde min max
        minarr = np.divide(self.min_array, self.num_of_activation)
        maxarr = np.divide(self.max_array, self.num_of_activation)

        # Claculate the ranges
        old_range = np.subtract(maxarr, minarr)
        new_range = np.subtract(new_max_arr, new_min_arr)
        rang = np.divide(new_range, old_range)

        # Normalize the data
        data = np.subtract(data, np.transpose(minarr))
        data = np.multiply(data, np.transpose(rang))
        data = np.add(data, np.transpose(new_min_arr))

        return data
