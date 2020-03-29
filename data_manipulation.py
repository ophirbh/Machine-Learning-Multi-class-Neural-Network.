import numpy as np


def add_bias(mat):
    for line in mat:
        line.append(1)
    return mat


def convert_to_float(vec):
    for i in range(len(vec)):
        vec[i, 0] = vec[i, 0].astype(float)

    return vec


def min_max_normalization(mat, new_min_arr, new_max_arr):
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

    return mat


def z_score_normalization(mat):
    mat = mat.astype(float)

    mean = np.mean(mat, 0)
    stand_dev = np.std(mat, 0)

    # If one of the parameters contains only one value avoid zero range
    for i in range(0, len(mat)):
        if stand_dev[i] == 0:
            mat[0, i] = mean[i] + 1

    # Recalculate the stand diviation after the change was made to the data
    # in order to avoid zero range
    original_stand_dev = stand_dev
    stand_dev = np.amax(mat, 0)

    for i in range(0, len(mat[0])):
        mat[i] = np.subtract(mat[i], np.transpose(mean))
        mat[i] = np.divide(mat[i], np.transpose(stand_dev))

    # If one of the parameters contains only one value fix the value was change to avoid zero range
    for i in range(0, len(mat)):
        if original_stand_dev[i] == 0:
            mat[:, i] = mean[i]

    return mat


def shuffle_data(data, labels):
    # Shuffle the data
    zipdata = list(zip(data, labels))
    np.random.shuffle(zipdata)
    data, labels = zip(*zipdata)

    data = np.asarray(data)
    data = data.astype(float)
    labels = np.asarray(labels)
    labels = labels.astype(float)

    return data, labels


def softmax(vec):
    normalize = np.subtract(vec, np.max(vec))
    numerator = np.exp(normalize)
    denominator = np.sum(numerator)
    softmax_vec = np.divide(numerator, denominator)
    return softmax_vec


def print_arr(arr):
    line = ""
    for value in arr:
        line = line + str(value) + "   "
    print(line)


def row_vec_multiply(vec1, vec2):
    mat = np.ones((len(vec1), len(vec2)))

    for index in range(len(vec1)):
        mat[index] = np.multiply(mat[index], vec1[index])

    mat = np.transpose(mat)

    for index in range(len(vec2)):
        mat[index] = np.multiply(mat[index], vec2[index])

    mat = np.transpose(mat)
    return mat

'''
arr = [[1, 2, 3, 4],
       [1, 3, 4, 5],
       [1, 5, 7, 9]]
arr = np.asarray(arr)
min_range = np.ones(4)
min_range = np.multiply(-1, min_range)
max_range = np.ones(4)
arr = min_max_normalization(arr, min_range, max_range)
print(arr)
a = np.ones(5)
b = np.ones(3)

for i in range(5):
    a[i] = (i + 1) * a[i]

for i in range(3):
    b[i] = b[i] * (i + 1)

print(a)
print(b)
print(row_vec_multiply(a, b))'''