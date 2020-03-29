import numpy as np
import data_manipulation as dm
import smart_min_max as norm

class NeuralNet(object):
    '''
    def __init__(self, train_x, activation_function, num_of__hidden_layers=1, num_of_nurons_per_layer=[16],  num_of_classes=9):
        self.num_of__hidden_layers = num_of__hidden_layers
        self.num_of_classes = num_of_classes
        self.activation_function = activation_function
        self.num_of_neurons_per_layer = num_of_nurons_per_layer
        # Create the weights in each layer of the neurons network.
        self.weights = []
        # Create each layer
        # First layer : the input size in the number of futures the data has  - rows of the matrix
        #         : the output size is the number of neurons in the first layer - columns of the matrix
        self.weights.append(np.zeros(np.rand(train_x.shape[1]), self.num_of_neurons_per_layer[0]))

        # Middle layers : the input is the number of neurons in the previous layer - rows of the matrix
        #               : the output is the number of neurons in the current layer - columns if the matrix
        for i in range(1, self.num_of__hidden_layers):
            self.weights.append(np.rand(self.num_of_neurons_per_layer[i - 1]), self.num_of_neurons_p]r_layer[i])

        # Last layer : the input is the number of neurons in the previous layer - rows of the matrix
        #            : the output is the number of classes - columns of the matrix
        self.weights.append(np.rand(self.self.num_of_neurons_per_layer[-1]), self.num_of_classes) '''

    def __init__(self, train_x, activation_function, eta=0.1, num_of_nurons_in_hidden_layer=16, num_of_classes=10):
        self.num_of_classes = num_of_classes
        self.activation_function = activation_function
        self.num_of_nurons_in_hidden_layer = num_of_nurons_in_hidden_layer
        self.eta = eta
        self.normalizetion = norm.smart_min_max(num_of_nurons_in_hidden_layer)
        self.loging = []


        # Create the weights in each layer of the neurons network.
        # Create each layer
        # First layer : the input size in the number of futures the data has  - rows of the matrix
        #         : the output size is the number of neurons in the first layer - columns of the matrix
        self.w1 = np.random.rand(train_x.shape[1], self.num_of_nurons_in_hidden_layer)
        self.bias1 = np.zeros(self.num_of_nurons_in_hidden_layer)

        # Second layer : the input is the number of neurons in the previous layer - rows of the matrix
        #            : the output is the number of classes - columns of the matrix
        self.w2 = np.random.rand(self.num_of_nurons_in_hidden_layer, self.num_of_classes)
        self.bias2 = np.zeros(self.num_of_classes)

    def initiate_weights(self, data):
        # Create the weights in each layer of the neurons network.
        # Create each layer
        # First layer : the input size in the number of futures the data has  - rows of the matrix
        #         : the output size is the number of neurons in the first layer - columns of the matrix
        self.w1 = np.random.uniform(-0.08, 0.08, [self.num_of_nurons_in_hidden_layer, data.shape[1]])
        self.bias1 = np.random.rand(self.num_of_nurons_in_hidden_layer, 1)

        # Second layer : the input is the number of neurons in the previous layer - rows of the matrix
        #            : the output is the number of classes - columns of the matrix
        self.w2 = np.random.uniform(-0.08, 0.08, [self.num_of_classes, self.num_of_nurons_in_hidden_layer])
        self.bias2 = np.random.rand(self.num_of_classes, 1)

    def predict(self, data, label):
        # Calculate the values through the net's hidden layers

        # Working with batch
        first_layer = np.zeros((len(data), self.num_of_nurons_in_hidden_layer))
        first_layer_after_activation = np.zeros((len(data), self.num_of_nurons_in_hidden_layer))
        second_layer = np.zeros((len(data), self.num_of_classes))
        y_hat = np.zeros((len(data), self.num_of_classes))
        prediction = np.zeros(len(data))
        loss = np.zeros(len(data))

        # First layer - foreach data in the batch - do the calculation of the first layer of the net
        for i in range(data.shape[0]):
            first_layer[i] = np.dot(data[i], self.w1)
            first_layer[i] = np.add(first_layer[i], self.bias1)

        # Normalization - do the normalization for all the batch together
        min_range = np.ones(first_layer.shape[1])
        min_range = np.multiply(-1, min_range)
        max_range = np.ones(first_layer.shape[1])
        first_layer = self.normalizetion.normalize(first_layer, min_range, max_range)

        # Activation function
        for i in range(data.shape[0]):
            first_layer_after_activation[i] = self.activation_function.calc(first_layer[i])

        # Second layer - foreach data in the batch - do the calculation of the second layer of the net
        for i in range(data.shape[0]):
            second_layer[i] = np.dot(first_layer_after_activation[i], self.w2)
            second_layer[i] = np.add(second_layer[i], self.bias2)
            y_hat[i] = dm.softmax(second_layer[i])

            # Get the class that got the biggest grade
            prediction[i] = np.argmax(y_hat[i])

            # Calc the loss
            loss[i] = self.cost(label[i], y_hat[i])

        # Back propagation
        # Calculate the negative gradient for the second set of weights
        # foreach data in the batch - do the calculation of the back propagation
        for i in range(data.shape[0]):
            y = np.zeros(self.num_of_classes)
            y[label[i].astype(int)] = 1
            # dloss/bias2
            dloss_bias2 = np.subtract(y, y_hat[i])
            dloss_bias2 = np.multiply(dloss_bias2, -1)

            # dloss/dw2
            dloss_dw2 = dm.row_vec_multiply(first_layer_after_activation[i], dloss_bias2)

            # Calculate the negative gradient for the first set of weights
            # dloss/bias1
            dloss_bias1 = np.subtract(y, y_hat[i])
            dloss_bias1 = np.multiply(dloss_bias1, -1)
            dloss_bias1 = np.dot(self.w2, dloss_bias1)
            dloss_bias1 = np.multiply(dloss_bias1, self.activation_function.calc_gradient(first_layer[i]))

            # dloss/dw1
            dloss_dw1 = dm.row_vec_multiply(data[i], dloss_bias1)

            # Update all the weights
            self.bias1 = np.subtract(self.bias1, np.multiply(self.eta, dloss_bias1))
            self.bias2 = np.subtract(self.bias2, np.multiply(self.eta, dloss_bias2))
            self.w1 = np.subtract(self.w1, np.multiply(self.eta, dloss_dw1))
            self.w2 = np.subtract(self.w2, np.multiply(self.eta, dloss_dw2))

        return prediction

    def predict_no_batch(self, data, label, update_flag=1):
        data = data.reshape(784, 1)

        # First layer
        first_layer = np.dot(self.w1, data)
        first_layer = np.add(first_layer, self.bias1)

        # Activation function
        first_layer_after_activation = self.activation_function.calc(first_layer)

        # Second layer
        second_layer = np.dot(self.w2, first_layer_after_activation)
        second_layer = np.add(second_layer, self.bias2)
        y_hat = dm.softmax(second_layer)

        # Get the class that got the biggest grade
        prediction = np.argmax(y_hat)

        if update_flag == 0:
            return prediction

        # Back propagation
        # Calculate the negative gradient for the second set of weights
        # foreach data in the batch - do the calculation of the back propagation
        y = np.zeros((self.num_of_classes, 1))
        y[label.astype(int)] = 1
        # dloss/bias2
        dloss_bias2 = np.subtract(y_hat, y)

        # dloss/dw2
        #dloss_dw2 = dm.row_vec_multiply(dloss_bias2, first_layer_after_activation)
        dloss_dw2 = np.dot(dloss_bias2, first_layer_after_activation.T)

        # Calculate the negative gradient for the first set of weights
        # dloss/bias1
        dloss_bias1 = np.subtract(y_hat, y)
        dloss_bias1 = np.dot(self.w2.T, dloss_bias1)
        dloss_bias1 = np.multiply(self.activation_function.calc_gradient(first_layer), dloss_bias1)

        # dloss/dw1
        dloss_dw1 = np.dot(dloss_bias1, data.T.reshape(1, 784))

        # Update all the weights
        self.bias1 = np.subtract(self.bias1, np.multiply(self.eta, dloss_bias1))
        self.bias2 = np.subtract(self.bias2, np.multiply(self.eta, dloss_bias2))
        self.w1 = np.subtract(self.w1, np.multiply(self.eta, dloss_dw1))
        self.w2 = np.subtract(self.w2, np.multiply(self.eta, dloss_dw2))

        return prediction

    def calc_accuracy(self, train_x, train_y, test_x, test_y):
        accuracy_train = 0
        accuracy_test = 0
        counter_train = 0
        counter_test = 0

        train_x = train_x.astype(float)
        train_y = train_y.astype(float)
        test_x = test_x.astype(float)
        test_y = test_y.astype(float)

        # Calc the accuracy on the data set
        for data_instance, label_instance in zip(train_x, train_y):
            predicted_label = self.predict_no_batch(data_instance, 0, 0)

            #print(str(predicted_label) + ":" + str(label_instance))

            if label_instance == predicted_label:
                counter_train += 1

        if len(train_x) != 0:
            accuracy_train = counter_train / len(train_x)

        for data_instance, label_instance in zip(test_x, test_y):
            predicted_label = self.predict_no_batch(data_instance, 0, 0)
            #print(str(predicted_label) + ":" + str(label_instance))

            if label_instance == predicted_label:
                counter_test += 1

        if len(test_x) != 0:
            accuracy_test = counter_test / len(test_x)

        counter_total = counter_train + counter_test
        if len(test_x) + len(train_x) != 0:
            accuracy_total = counter_total / (len(test_x) + len(train_x))

        return accuracy_total, accuracy_test, accuracy_train

    def k_fold_data_arrnge(self, data, labels, number_of_parts, test_part_number):
        size = len(data)
        if number_of_parts == 1:
            data_x = data
            data_y = labels
            test_x = np.zeros(data.shape[1])
            test_y = np.zeros(labels.shape[1])
        elif test_part_number == 0:
            test_x = data[0: int(size / number_of_parts)]
            test_y = labels[0: int(size / number_of_parts)]
            data_x = data[int(size / number_of_parts): size]
            data_y = labels[int(size / number_of_parts): size]
        elif test_part_number == number_of_parts - 1:
            test_x = data[int(size * (number_of_parts - 1) / number_of_parts): size]
            test_y = labels[int(size * (number_of_parts - 1) / number_of_parts): size]
            data_x = data[0: int(size * (number_of_parts - 1) / number_of_parts)]
            data_y = labels[0:int(size * (number_of_parts - 1) / number_of_parts)]
        else:
            test_x = data[int(size * test_part_number / number_of_parts):
                          int(size * (test_part_number + 1) / number_of_parts)]
            test_y = labels[int(size * test_part_number / number_of_parts):
                            int(size * (test_part_number + 1) / number_of_parts)]

            pretest_x = data[0: int(size * test_part_number / number_of_parts)]
            posttest_x = data[int(size * (test_part_number + 1) / number_of_parts): size]
            pretest_y = labels[0: int(size * test_part_number / number_of_parts)]
            posttest_y = labels[int(size * (test_part_number + 1) / number_of_parts): size]

            data_x = np.concatenate((pretest_x, posttest_x), 0)
            data_y = np.concatenate((pretest_y, posttest_y), 0)

        return data_x, data_y, test_x, test_y

    def cost(self, y, y_hat):
        return 0

    def shuffle_data(self, data, labels):
        # Shuffle the data
        zipdata = list(zip(data, labels))
        np.random.shuffle(zipdata)
        data, labels = zip(*zipdata)

        data = np.asarray(data)
        data = data.astype(float)
        labels = np.asarray(labels)
        labels = labels.astype(float)

        return data, labels

    def train(self, data, labels, epoch_num, k_fold_parameter=1):
        # Shuffle the data
        data, labels = self.shuffle_data(data, labels)

        # Accuracy variable
        avg_accuracy = 0

        for i in range(k_fold_parameter):
            # Initiate the weights
            self.initiate_weights(data)

            data_x, data_y, test_x, test_y = self.k_fold_data_arrnge(data, labels, k_fold_parameter, i)

            for epoch in range(epoch_num):
                print(".")
                for j in range(data_x.shape[0]):
                    self.predict_no_batch(data_x[j], data_y[j])

            if k_fold_parameter != 1:
                # Check the accuracy of the current k fold test part
                accuracy_total, accuracy_test, accuracy_train = self.calc_accuracy(data_x, data_y, test_x, test_y)
                avg_accuracy += accuracy_total

        # If this is not testing mode
        if k_fold_parameter == 1:
            return 0

        avg_accuracy = avg_accuracy / k_fold_parameter
        return avg_accuracy
