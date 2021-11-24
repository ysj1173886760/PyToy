# coding:utf-8
import numpy as np
import cupy as cp
import struct
import os
import scipy.io
import time
import tqdm

from operators import BatchNormLayer, FullyConnectedLayer, ReLULayer, SoftmaxLossLayer, ConvolutionalLayer, MaxPoolingLayer, FlattenLayer

class Network(object):
    def __init__(self, param_path=''):
        self.param_path = param_path
        self.param_layer_name = [
            'conv1_1', 'bn1', 'relu1_2', 'pool1',  
            'conv2_1', 'bn2', 'relu2_2', 'pool2', 
            'conv3_1', 'bn3', 'relu3_2', 'pool3', 
            'flatten', 'fc1', 'fc2', 'softmax'
        ]

    def build_model(self):
        print('Building model...')

        self.layers = {}

        # 32 * 32 * 3
        self.layers['conv1_1'] = ConvolutionalLayer(3, 3, 32, 1, 1, 0.01)
        self.layers['bn1'] = BatchNormLayer((32, 32, 32))
        self.layers['relu1_2'] = ReLULayer()
        self.layers['pool1'] = MaxPoolingLayer(2, 2)

        self.layers['conv2_1'] = ConvolutionalLayer(3, 32, 32, 1, 1, 0.01)
        self.layers['bn2'] = BatchNormLayer((32, 16, 16))
        self.layers['relu2_2'] = ReLULayer()
        self.layers['pool2'] = MaxPoolingLayer(2, 2)

        self.layers['conv3_1'] = ConvolutionalLayer(3, 32, 64, 1, 1, 0.01)
        self.layers['bn3'] = BatchNormLayer((64, 8, 8))
        self.layers['relu3_2'] = ReLULayer()
        self.layers['pool3'] = MaxPoolingLayer(2, 2)

        self.layers['flatten'] = FlattenLayer((64, 4, 4), (1024, ))
        self.layers['fc1'] = FullyConnectedLayer(1024, 1024, 0.1)
        self.layers['fc2'] = FullyConnectedLayer(1024, 10, 0.1)

        self.layers['softmax'] = SoftmaxLossLayer()

        self.update_layer_list = []
        for layer_name in self.layers.keys():
            if ('conv' in layer_name) or ('fc' in layer_name) or ('bn' in layer_name):
                self.update_layer_list.append(layer_name)

    def init_model(self):
        print('Initializing parameters of each layer...')
        for layer_name in self.update_layer_list:
            self.layers[layer_name].init_param()

    def load_model(self):
        pass

    def forward(self, input_image, train=True):
        # start_time = time.time()
        current = input_image
        for idx in range(len(self.param_layer_name)):
            # TODO： 计算VGG19网络的前向传播
            current = self.layers[self.param_layer_name[idx]].forward(current, train)
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

    def evaluate(self, test_data, test_label):
        test = cp.array(test_data)
        label = cp.array(test_label)
        pred_results = cp.zeros([test.shape[0]])
        for idx in range(int(test.shape[0] / BATCH_SIZE)):
            batch_images = test[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
            prob = self.forward(batch_images, False)
            pred_labels = cp.argmax(prob, axis=1)
            pred_results[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE] = pred_labels
        accuracy = cp.mean(pred_results == label)
        return accuracy

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == '__main__':
    TRAIN_STEP = 100
    LEARNING_RATE = 0.1
    BATCH_SIZE = 100
    # data_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    data_list = ['data_batch_1']

    net = Network()
    net.build_model()
    net.init_model()
    
    # load train data
    for i, pth in enumerate(data_list):
        data_pth = os.path.join('./data', pth)
        data = unpickle(data_pth)
        images = np.transpose(data[b'data'].reshape(-1, 32, 32, 3), [0, 3, 1, 2])
        labels = np.array(data[b'labels']).reshape(-1)
        if i == 0:
            train_data = images
            train_label = labels
        else:
            train_data = np.concatenate((train_data, images))
            train_label = np.concatenate((train_label, labels))
    
    # load test data
    test_data_raw = unpickle(os.path.join('./data', 'test_batch'))
    test_data = np.transpose(test_data_raw[b'data'].reshape(-1, 32, 32, 3), [0, 3, 1, 2])
    test_label = np.array(test_data_raw[b'labels']).reshape(-1)
    random_index = np.arange(train_data.shape[0]).astype(int)
    max_batch = train_data.shape[0] // BATCH_SIZE
    last_accuracy = 0.0

    # tqdm stuff
    for epoch in range(TRAIN_STEP):
        np.random.shuffle(random_index)
        train_data = train_data[random_index]
        train_label = train_label[random_index]
        bar = tqdm.tqdm(range(max_batch))
        for cur in bar:
            batch_image = cp.array(train_data[cur * BATCH_SIZE: (cur + 1) * BATCH_SIZE])
            batch_label = cp.array(train_label[cur * BATCH_SIZE: (cur + 1) * BATCH_SIZE])
            prob = net.forward(batch_image)
            loss = net.layers['softmax'].get_loss(batch_label)
            net.backward(loss)
            net.update(LEARNING_RATE)
            bar.set_description("Epoch %d Loss %.6f Accuracy %.3f" % (epoch, loss, last_accuracy))
            # print("batch time %f" % (end_time - start_time))
            
        last_accuracy = net.evaluate(test_data, test_label)
        

