# -*- encoding: utf-8 -*-
'''
@File    :   mnist.py
@Time    :   2021/11/30 15:08:31
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   None
'''

import numpy as cp
import struct
import os
import time
import pytoy as pt
from pytoy.ops.ops import SoftMax

MNIST_DIR = "./mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"

def computeMse(input1, input2):
    return cp.sum(cp.square(input1.flatten() - input2.flatten()))

class MNIST_MLP(object):
    def __init__(self, batch_size=100, input_size=784, hidden1=32, hidden2=16, out_classes=10, lr=0.01, max_epoch=1, print_iter=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter

    def load_mnist(self, file_dir, is_images = 'True'):
        # Read binary data
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()
        # Analysis file header
        if is_images:
            # Read images
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:
            # Read labels
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = cp.reshape(mat_data, [num_images, num_rows * num_cols])
        print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))
        return mat_data

    def load_data(self):
        print('Loading MNIST data from files...')
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)
        self.train_data = cp.append(train_images, train_labels, axis=1)
        self.test_data = cp.append(test_images, test_labels, axis=1)

    def shuffle_data(self):
        print('Randomly shuffle MNIST data...')
        cp.random.shuffle(self.train_data)

    def build_model(self):  # 建立网络结构
        print('Building multi-layer perception model...')
        self.input = pt.core.Variable((self.batch_size, self.input_size), init=False, trainable=False)
        self.fc1 = pt.layer.Dense(self.input, self.input_size, self.hidden1, std=0.01, name='fc1')
        self.relu1 = pt.ops.ReLU(self.fc1)

        self.fc2 = pt.layer.Dense(self.relu1, self.hidden1, self.hidden2, std=0.01, name='fc2')
        self.relu2 = pt.ops.ReLU(self.fc2)

        self.fc3 = pt.layer.Dense(self.relu2, self.hidden2, self.out_classes, std=0.01, name='fc3')
        self.relu3 = pt.ops.ReLU(self.fc3)

        self.softmax = pt.ops.SoftMax(self.relu3)
        self.label = pt.core.Variable((self.batch_size, ), init=False, trainable=False)
        self.loss = pt.loss.CrossEntropyWithSoftMax(self.relu3, self.label)
    
        # correct version
        # self.t_fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        # self.t_relu1 = ReLULayer()

        # self.t_fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        # self.t_relu2 = ReLULayer()

        # self.t_fc3 = FullyConnectedLayer(self.hidden2, self.out_classes)
        # self.t_relu3 = ReLULayer()

        # self.t_softmax = SoftmaxLossLayer()
        # self.update_layer_list = [self.t_fc1, self.t_fc2, self.t_fc3]
        # for layer in self.update_layer_list:
        #     layer.init_param()



    def train(self):
        max_batch = self.train_data.shape[0] // self.batch_size
        # print('before training')
        # self.evaluate()
        # print('Start training...')
        trainable_node = pt.core.get_trainable_variables_from_graph()
        # pt.get_node_from_graph('Variable:1').set_value(self.t_fc1.weight)
        # pt.get_node_from_graph('Variable:2').set_value(self.t_fc1.bias)

        # pt.get_node_from_graph('Variable:6').set_value(self.t_fc2.weight)
        # pt.get_node_from_graph('Variable:7').set_value(self.t_fc2.bias)

        # pt.get_node_from_graph('Variable:11').set_value(self.t_fc3.weight)
        # pt.get_node_from_graph('Variable:12').set_value(self.t_fc3.bias)
        pt.default_graph.draw()

        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            for idx_batch in range(max_batch):
                batch_images = self.train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, :-1]
                batch_labels = self.train_data[idx_batch*self.batch_size:(idx_batch+1)*self.batch_size, -1]
                self.input.set_value(batch_images)
                self.label.set_value(batch_labels)
                self.loss.forward()
                
                # r_fc1 = self.t_fc1.forward(batch_images)
                # r_relu1 = self.t_relu1.forward(r_fc1)
                # r_fc2 = self.t_fc2.forward(r_relu1)
                # r_relu2 = self.t_relu2.forward(r_fc2)
                # r_fc3 = self.t_fc3.forward(r_relu2)
                # r_relu3 = self.t_relu3.forward(r_fc3)
                # prob = self.t_softmax.forward(r_relu3)
                # loss2 = self.t_softmax.get_loss(batch_labels)

                # assert (r_fc1 == pt.get_node_from_graph('Add:4').value).all()
                # assert (r_fc2 == pt.get_node_from_graph('Add:9').value).all()
                # assert (r_fc3 == pt.get_node_from_graph('Add:14').value).all()
                # assert (prob == self.loss.prob).all()
                
                # dloss = self.t_softmax.backward()
                # b_relu3 = self.t_relu3.backward(dloss)
                # b_fc3 = self.t_fc3.backward(b_relu3)
                # b_relu2 = self.t_relu2.backward(b_fc3)
                # b_fc2 = self.t_fc2.backward(b_relu2)
                # b_relu1 = self.t_relu1.backward(b_fc2)
                # b_fc1 = self.t_fc1.backward(b_relu1)

                for node in trainable_node:
                    node.backward(self.loss)

                # assert computeMse(b_relu3, pt.get_node_from_graph('Add:14').graident) < 1e-10
                # assert computeMse(b_relu2, pt.get_node_from_graph('Add:9').graident) < 1e-10
                # assert computeMse(b_relu1, pt.get_node_from_graph('Add:4').graident) < 1e-10
                # assert computeMse(self.t_fc3.d_weight, pt.get_node_from_graph('Variable:11').graident) < 1e-10
                # assert computeMse(self.t_fc3.d_bias, pt.get_node_from_graph('Variable:12').graident) < 1e-10
                # assert computeMse(self.t_fc2.d_weight, pt.get_node_from_graph('Variable:6').graident) < 1e-10
                # assert computeMse(self.t_fc2.d_bias, pt.get_node_from_graph('Variable:7').graident) < 1e-10
                # assert computeMse(self.t_fc1.d_weight, pt.get_node_from_graph('Variable:1').graident) < 1e-10
                # assert computeMse(self.t_fc1.d_bias, pt.get_node_from_graph('Variable:2').graident) < 1e-10
                # for layer in self.update_layer_list:
                #     layer.update_param(self.lr)

                loss1 = self.loss.value

                for node in trainable_node:
                    node.set_value(node.value - self.lr * node.graident)

                pt.default_graph.clear_graident()

                if idx_batch % self.print_iter == 0:
                    print('Epoch %d, iter %d, loss1: %.6f loss2: %.6f' % (idx_epoch, idx_batch, loss1, loss2))

            self.evaluate1()
            self.evaluate2()

    def evaluate1(self):
        pred_results = cp.zeros([self.test_data.shape[0]])
        total_time = 0
        for idx in range(self.test_data.shape[0] // self.batch_size):
            batch_images = self.test_data[idx*self.batch_size:(idx+1)*self.batch_size, :-1]
            start = time.time()
            self.input.set_value(batch_images)
            self.softmax.forward()
            end = time.time()
            total_time += (end - start)
            pred_labels = cp.argmax(self.softmax.value, axis=1)
            pred_results[idx*self.batch_size:(idx+1)*self.batch_size] = pred_labels
        accuracy = cp.mean(pred_results == self.test_data[:,-1])
        print("inferencing time: %f"% (total_time))
        print('Accuracy in test set: %f' % accuracy)

    def evaluate2(self):
        pred_results = cp.zeros([self.test_data.shape[0]])
        total_time = 0
        for idx in range(self.test_data.shape[0] // self.batch_size):
            batch_images = self.test_data[idx*self.batch_size:(idx+1)*self.batch_size, :-1]
            start = time.time()
            r_fc1 = self.t_fc1.forward(batch_images)
            r_relu1 = self.t_relu1.forward(r_fc1)
            r_fc2 = self.t_fc2.forward(r_relu1)
            r_relu2 = self.t_relu2.forward(r_fc2)
            r_fc3 = self.t_fc3.forward(r_relu2)
            r_relu3 = self.t_relu3.forward(r_fc3)
            prob = self.t_softmax.forward(r_relu3)
            end = time.time()
            total_time += (end - start)
            pred_labels = cp.argmax(prob, axis=1)
            pred_results[idx*self.batch_size:(idx+1)*self.batch_size] = pred_labels
        accuracy = cp.mean(pred_results == self.test_data[:,-1])
        print("inferencing time: %f"% (total_time))
        print('Accuracy in test set: %f' % accuracy)

def build_mnist_mlp(param_dir='weight.npy'):
    h1, h2, e = 32, 16, 10
    mlp = MNIST_MLP(hidden1=h1, hidden2=h2, max_epoch=e)
    mlp.load_data()
    mlp.build_model()
    mlp.train()
    return mlp

if __name__ == '__main__':
    mlp = build_mnist_mlp()