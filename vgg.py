# coding:utf-8
import numpy as np
import struct
import os
import scipy.io
import time

from operators import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer, ConvolutionalLayer, MaxPoolingLayer, FlattenLayer

class Network(object):
    def __init__(self, param_path=''):
        self.param_path = param_path
        self.param_layer_name = [
            'conv1_1', 'relu1_2', 'pool1', 
            'conv2_1', 'relu2_2', 'pool2', 
            'conv3_1', 'relu3_2', 'pool3', 
            'flatten', 'fc6', 'softmax'
        ]

    def build_model(self):
        print('Building model...')

        self.layers = {}

        # 32 * 32 * 3
        self.layers['conv1_1'] = ConvolutionalLayer(3, 3, 32, 1, 1, 0.0001)
        self.layers['relu1_2'] = ReLULayer()
        self.layers['pool1'] = MaxPoolingLayer(2, 2)
        self.layers['conv2_1'] = ConvolutionalLayer(3, 32, 32, 1, 1, 0.01)
        self.layers['relu2_2'] = ReLULayer()
        self.layers['pool2'] = MaxPoolingLayer(2, 2)

        self.layers['conv3_1'] = ConvolutionalLayer(3, 32, 64, 1, 1, 0.01)
        self.layers['relu3_2'] = ReLULayer()
        self.layers['pool3'] = MaxPoolingLayer(2, 2)

        self.layers['flatten'] = FlattenLayer((64, 4, 4), (1024, ))
        self.layers['fc6'] = FullyConnectedLayer(1024, 10, 0.1)

        self.layers['softmax'] = SoftmaxLossLayer()

        self.update_layer_list = []
        for layer_name in self.layers.keys():
            if 'conv' in layer_name or 'fc' in layer_name:
                self.update_layer_list.append(layer_name)

    def init_model(self):
        print('Initializing parameters of each layer...')
        for layer_name in self.update_layer_list:
            self.layers[layer_name].init_param()

    def load_model(self):
        pass

    def forward(self, input_image):
        # start_time = time.time()
        current = input_image
        for idx in range(len(self.param_layer_name)):
            # TODO： 计算VGG19网络的前向传播
            current = self.layers[self.param_layer_name[idx]].forward(current)
        # print('Forward time: %f' % (time.time()-start_time))
        return current

    def backward(self, dloss):
        # start_time = time.time()
        for idx in range(len(self.param_layer_name) - 1, -1, -1):
            dloss = self.layers[self.param_layer_name[idx]].backward(dloss)

        #print('Backward time: %f' % (time.time()-start_time))
        return dloss

    def update(self, lr):
        for layer_name in self.update_layer_list:
            self.layers[layer_name].update_param(lr)

    def evaluate(self, test_data):
        np.random.shuffle(test_data)
        test = test_data[0: 100, :]
        pred_results = np.zeros([test.shape[0]])
        total_time = 0
        for idx in range(int(test.shape[0] / BATCH_SIZE)):
            batch_images = test[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE, : -1].reshape(-1, 3, 32, 32)
            start = time.time()
            prob = self.forward(batch_images)
            end = time.time()
            total_time += (end - start)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE] = pred_labels
        accuracy = np.mean(pred_results == test[:,-1])
        print("inferencing time: %f"% (total_time))
        print('Accuracy in test set: %f' % accuracy)

class AdamOptimizer(object):
    def __init__(self, lr):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.lr = lr
        self.step = 0

    def update(self, input, grad):
        self.step += 1
        self.mt = self.beta1 * self.mt + (1 - self.beta1) * grad
        self.vt = self.beta2 * self.vt + (1 - self.beta2) * (grad ** 2)
        mt_hat = self.mt / (1 - np.power(self.beta1, self.step))
        vt_hat = self.vt / (1 - np.power(self.beta2, self.step))
        output = input - self.lr * mt_hat / (np.sqrt(vt_hat) + self.eps)
        return output

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == '__main__':
    TRAIN_STEP = 100
    LEARNING_RATE = 0.1
    BATCH_SIZE = 100
    PRINT_ITER = 10
    # data_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    data_list = ['data_batch_1']

    net = Network()
    net.build_model()
    net.init_model()
    adam_optimizer = AdamOptimizer(LEARNING_RATE)
    
    # load train data
    for i, pth in enumerate(data_list):
        data_pth = os.path.join('./data', pth)
        data = unpickle(data_pth)
        images = np.transpose(data[b'data'].reshape(-1, 32, 32, 3), [0, 3, 1, 2]).reshape(-1, 32 * 32 * 3)
        labels = np.array(data[b'labels']).reshape(-1, 1)
        images = np.hstack((images, labels))
        if i == 0:
            train_data = images
        else:
            train_data = np.concatenate((train_data, images))
    
    # load test data
    test_data = unpickle(os.path.join('./data', 'test_batch'))
    test_data = np.hstack((np.transpose(test_data[b'data'].reshape(-1, 32, 32, 3), [0, 3, 1, 2]).reshape(-1, 32 * 32 * 3), np.array(test_data[b'labels']).reshape(-1, 1)))
    max_batch = int(train_data.shape[0] / BATCH_SIZE)

    for epoch in range(TRAIN_STEP):
        np.random.shuffle(train_data)
        for cur in range(max_batch):
            start_time = time.time()
            batch_data = train_data[cur * BATCH_SIZE: cur * BATCH_SIZE + BATCH_SIZE]
            batch_image = batch_data[:, 0: -1].reshape(-1, 3, 32, 32)
            batch_label = batch_data[:, -1]
            prob = net.forward(batch_image)
            loss = net.layers['softmax'].get_loss(batch_label)
            net.backward(loss)
            net.update(LEARNING_RATE)
            end_time = time.time()
            # print("batch time %f" % (end_time - start_time))
            if cur % PRINT_ITER == 0:
                print('Epoch %d, iter %d, loss: %.6f' % (epoch, cur, loss))
                net.evaluate(test_data)
        

