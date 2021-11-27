
import cProfile
import numpy as np
import cupy as cp
import time
import tqdm
import os

from operators import BatchNormLayer, DropOut, FullyConnectedLayer, ReLULayer, SoftmaxLossLayer, ConvolutionalLayer, MaxPoolingLayer, FlattenLayer

TRAIN_STEP = 100
BATCH_SIZE = 100
SAVE_EPOCH = 1
MODEL_DIR = './models'

class Network(object):
    def __init__(self, lr=1, optimizer=False):
        self.optimizer = optimizer
        self.lr = lr

    def build_model(self, model):
        print('Building model...')
        self.param_layer_name = model['layer_name']
        self.layers = model['layers']

        self.update_layer_list = []
        for layer_name in self.layers.keys():
            if ('conv' in layer_name) or ('fc' in layer_name) or ('bn' in layer_name):
                self.update_layer_list.append(layer_name)

    def init_model(self):
        print('Initializing parameters of each layer...')
        for layer_name in self.update_layer_list:
            self.layers[layer_name].init_param(self.lr, self.optimizer)

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

    def update(self):
        for layer_name in self.update_layer_list:
            self.layers[layer_name].update_param()

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
    
    def save_model(self, param_dir):
        params = {}
        for layer_name in self.update_layer_list:
            params[layer_name] = self.layers[layer_name].get_param()
        np.save(param_dir, params)
    
    def load_model(self, param_dir):
        params = np.load(param_dir, allow_pickle=True).item()
        for layer_name in self.update_layer_list:
            self.layers[layer_name].load_param(params[layer_name])
        print("Loading Model Complete")
    
    def train(self, train_data, train_label, test_data, test_label):
        random_index = np.arange(train_data.shape[0]).astype(int)
        max_batch = train_data.shape[0] // BATCH_SIZE
        last_accuracy = 0.0
        train_accuracy = 0.0

        # tqdm stuff
        for epoch in range(TRAIN_STEP):
            np.random.shuffle(random_index)
            train_data = train_data[random_index]
            train_label = train_label[random_index]
            bar = tqdm.tqdm(range(max_batch))
            total_loss = 0
            for cur in bar:
                batch_image = cp.array(train_data[cur * BATCH_SIZE: (cur + 1) * BATCH_SIZE])
                batch_label = cp.array(train_label[cur * BATCH_SIZE: (cur + 1) * BATCH_SIZE])
                # cProfile.run("self.forward(batch_image)")
                prob = self.forward(batch_image)
                loss = self.layers['softmax'].get_loss(batch_label)
                total_loss += loss
                # cProfile.run("self.backward(loss)")
                self.backward(loss)
                # cProfile.run("self.update()")
                self.update()
                bar.set_description("Epoch %d Loss %.6f ValidationAccuracy %.3f TrainAccuracy %.3f" % (epoch, total_loss / (cur + 1), last_accuracy, train_accuracy))
                # print("batch time %f" % (end_time - start_time))
                
            last_accuracy = self.evaluate(test_data, test_label)
            train_accuracy = self.evaluate(train_data[0: 10000], train_label[0: 10000])

            if (epoch + 1) % SAVE_EPOCH == 0:
                self.save_model(os.path.join(MODEL_DIR, 'model{}.npy'.format(epoch)))

class lightWeightNetwork(object):
    def get_model(self):
        param_layer_name = [
            'conv1_1', 'bn1', 'relu1_2', 'pool1',  
            'conv2_1', 'bn2', 'relu2_2', 'pool2', 
            'conv3_1', 'bn3', 'relu3_2', 'pool3', 
            'flatten', 'fc1', 'softmax'
        ]

        layers = {}

        layers['conv1_1'] = ConvolutionalLayer(3, 3, 32, 1, 1, 0.1)
        layers['bn1'] = BatchNormLayer((32, 32, 32))
        layers['relu1_2'] = ReLULayer()
        layers['pool1'] = MaxPoolingLayer(2, 2)

        layers['conv2_1'] = ConvolutionalLayer(3, 32, 64, 1, 1, 0.1)
        layers['bn2'] = BatchNormLayer((64, 16, 16))
        layers['relu2_2'] = ReLULayer()
        layers['pool2'] = MaxPoolingLayer(2, 2)

        layers['conv3_1'] = ConvolutionalLayer(3, 64, 128, 1, 1, 0.01)
        layers['bn3'] = BatchNormLayer((128, 8, 8))
        layers['relu3_2'] = ReLULayer()
        layers['pool3'] = MaxPoolingLayer(2, 2)

        layers['flatten'] = FlattenLayer((128, 4, 4), (2048, ))
        layers['fc1'] = FullyConnectedLayer(2048, 10, 0.01)

        layers['softmax'] = SoftmaxLossLayer()

        model = {}
        model['layer_name'] = param_layer_name
        model['layers'] = layers

        return model

class DeeperNetwork(object):
    def get_model(self):
        param_layer_name = [
            'conv1_1', 'bn1', 'relu1_2', 'pool1',  
            'conv2_1', 'bn2', 'relu2_2', 'pool2', 
            'conv3_1', 'bn3', 'relu3_2', 'pool3', 
            'flatten', 'fc4_1', 'relu4_2', 'fc4_3', 'softmax'
        ]

        layers = {}

        layers['conv1_1'] = ConvolutionalLayer(3, 3, 64, 1, 1, 0.001)
        layers['bn1'] = BatchNormLayer((64, 32, 32))
        layers['relu1_2'] = ReLULayer()
        layers['pool1'] = MaxPoolingLayer(2, 2)

        layers['conv2_1'] = ConvolutionalLayer(3, 64, 128, 1, 1, 0.001)
        layers['bn2'] = BatchNormLayer((128, 16, 16))
        layers['relu2_2'] = ReLULayer()
        layers['pool2'] = MaxPoolingLayer(2, 2)

        layers['conv3_1'] = ConvolutionalLayer(3, 128, 256, 1, 1, 0.001)
        layers['bn3'] = BatchNormLayer((256, 8, 8))
        layers['relu3_2'] = ReLULayer()
        layers['pool3'] = MaxPoolingLayer(2, 2)

        layers['flatten'] = FlattenLayer((256, 4, 4), (4096, ))
        layers['fc4_1'] = FullyConnectedLayer(4096, 1024, 0.001)
        layers['relu4_2'] = ReLULayer()
        # layers['dropout'] = DropOut(0.4)
        layers['fc4_3'] = FullyConnectedLayer(1024, 10, 0.001)

        layers['softmax'] = SoftmaxLossLayer()
        model = {}
        model['layer_name'] = param_layer_name
        model['layers'] = layers

        return model