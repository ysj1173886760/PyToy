import pytoy as pt
import numpy as np

from pytoy.layer.layer import Dense
from pytoy.core.node import Variable
from pytoy.ops.ops import SoftMax
from pytoy.ops.loss import CrossEntropyWithSoftMax
from pytoy.ops import *
from pytoy.layer.layer import RNN

max_len = 100
input_size = 16
hidden_size = 12
batch_size = 10

# get_sequence_data just for generate train set
from scipy import signal

def get_sequence_data(dimension=10, length=10, number_of_example=1000, train_set_ratio=0.7, seed=42):
    xx = []
    xx.append(np.sin(np.arange(0, 10, 10 / length)).reshape(-1, 1))
    xx.append(np.array(signal.square(np.arange(0, 10, 10 / length))).reshape(-1, 1))

    data = []
    for i in range(2):
        x = xx[i]
        for j in range(number_of_example // 2):
            sequence = x + np.random.normal(0, 0.6, (len(x), dimension))
            label = np.array([int(i == 0)])
            data.append(np.c_[sequence.reshape(1, -1), label.reshape(1, -1)])

    data = np.concatenate(data, axis=0)
    
    np.random.shuffle(data)

    train_set_size = int(number_of_example * train_set_ratio)

    return (data[:train_set_size, :-1].reshape(-1, length, dimension),
            data[:train_set_size, -1:],
            data[train_set_size:, :-1].reshape(-1, length, dimension),
            data[train_set_size:, -1:])

signal_train, label_train, signal_test, label_test = get_sequence_data(length=max_len, dimension=input_size)

inputs = [Variable(dims=(batch_size, input_size), init=False, trainable=False) for i in range(max_len)]
last_step = RNN(inputs, input_size, hidden_size)
output = Dense(last_step, hidden_size, 2)
predict = SoftMax(output)

label = Variable(dims=(batch_size, 1), trainable=False)
loss = CrossEntropyWithSoftMax(output, label)

learning_rate = 0.005
adam = pt.optimizer.Adam(pt.default_graph, loss, learning_rate)

for epoch in range(30):
    for i in range(0, len(signal_train), batch_size):
        # signal_train : (sample_number, max_len, input_size)
        # inputs : (max_len, (1, input_size))
        for j, iv in enumerate(inputs):
            iv.set_value(np.mat(signal_train[i:i + batch_size, j]))
        
        label.set_value(np.mat(label_train[i:i + batch_size]))
        adam.step()
        adam.update()

    print("epoch {:d} is over".format(epoch + 1))

    pred = []
    for i in range(0, len(signal_test), batch_size):
        for j, iv in enumerate(inputs):
            iv.set_value(np.mat(signal_test[i:i + batch_size, j]))

        predict.forward()
        pred.append(predict.value)

    pred = np.array(pred).argmax(axis=2)
    label_test = label_test.reshape(-1, batch_size)

    accuracy = (label_test == pred).sum() / len(signal_test)
    print("epoch: {:d}, accuracy: {:.5f}".format(epoch+1, accuracy))