import numpy as np
import neural_net as nn
import data_manipulation as dm
import relu

train_x = np.loadtxt("train_x", max_rows=20000) / 255
train_y = np.loadtxt("train_y", max_rows=20000)

activation = relu.relu()
net = nn.NeuralNet(train_x, activation, 0.01, 50, 10)
acuurcy = net.train(train_x, train_y, 10, 2)
print(acuurcy)

test_x = np.loadtxt("test_x") / 255
test_y = np.zeros(test_x.shape[0])

for i in range(test_x.shape[0]):
    test_y[i] = (net.predict_no_batch(test_x[i], 0, 0))

output = ""
for i in range(test_y.shape[0]):
    output = output + str(test_y[i].astype(int)) + "\n"

output_file = open("test_y", "w+")
output_file.write(output)
output_file.close()


