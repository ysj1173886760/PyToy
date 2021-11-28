
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
        total_loss = 0
        for idx in range(int(test.shape[0] / BATCH_SIZE)):
            batch_images = test[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
            prob = self.forward(batch_images, False)
            total_loss += self.layers['softmax'].get_loss(label[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE])
            pred_labels = cp.argmax(prob, axis=1)
            pred_results[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE] = pred_labels
        accuracy = cp.mean(pred_results == label)
        return accuracy, total_loss
    
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
        validation_loss = 0.0

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
                bar.set_description("Epoch %d Loss %.6f ValidationLoss %.3f ValidationAccuracy %.3f TrainAccuracy %.3f" % (epoch, total_loss / (cur + 1), validation_loss, last_accuracy, train_accuracy))
                # print("batch time %f" % (end_time - start_time))
                
            last_accuracy, validation_loss = self.evaluate(test_data, test_label)
            train_accuracy, _ = self.evaluate(train_data[0: 10000], train_label[0: 10000])

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
            'flatten', 'fc4_1', 'relu4_2', 'bn4', 'fc4_3', 'softmax'
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
        layers['bn4'] = BatchNormLayer((1024, ))
        layers['fc4_3'] = FullyConnectedLayer(1024, 10, 0.001)

        layers['softmax'] = SoftmaxLossLayer()
        model = {}
        model['layer_name'] = param_layer_name
        model['layers'] = layers

        return model

class testNetwork(object):
    def get_model(self):
        param_layer_name = [
            'conv1_1', 'relu1_2', 'bn1_3', 'conv1_4', 'relu1_5', 'bn1_6', 'pool1_7', 'dropout1_8',
            'conv2_1', 'relu2_2', 'bn2_3', 'conv2_4', 'relu2_5', 'bn2_6', 'pool2_7', 'dropout2_8',
            'conv3_1', 'relu3_2', 'bn3_3', 'conv3_4', 'relu3_5', 'bn3_6', 'pool3_7', 'dropout3_8',
            'flatten', 'fc4_1', 'relu4_2', 'bn4_3', 'dropout4_4', 'fc4_5', 'softmax'
        ]

        layers = {}

        layers['conv1_1'] = ConvolutionalLayer(3, 3, 32, 1, 1, 0.001)
        layers['relu1_2'] = ReLULayer()
        layers['bn1_3'] = BatchNormLayer((32, 32, 32))
        layers['conv1_4'] = ConvolutionalLayer(3, 32, 32, 1, 1, 0.001)
        layers['relu1_5'] = ReLULayer()
        layers['bn1_6'] = BatchNormLayer((32, 32, 32))
        layers['pool1_7'] = MaxPoolingLayer(2, 2)
        layers['dropout1_8'] = DropOut(0.3)

        layers['conv2_1'] = ConvolutionalLayer(3, 32, 64, 1, 1, 0.001)
        layers['relu2_2'] = ReLULayer()
        layers['bn2_3'] = BatchNormLayer((64, 16, 16))
        layers['conv2_4'] = ConvolutionalLayer(3, 64, 64, 1, 1, 0.001)
        layers['relu2_5'] = ReLULayer()
        layers['bn2_6'] = BatchNormLayer((64, 16, 16))
        layers['pool2_7'] = MaxPoolingLayer(2, 2)
        layers['dropout2_8'] = DropOut(0.5)

        layers['conv3_1'] = ConvolutionalLayer(3, 64, 128, 1, 1, 0.001)
        layers['relu3_2'] = ReLULayer()
        layers['bn3_3'] = BatchNormLayer((128, 8, 8))
        layers['conv3_4'] = ConvolutionalLayer(3, 128, 128, 1, 1, 0.001)
        layers['relu3_5'] = ReLULayer()
        layers['bn3_6'] = BatchNormLayer((128, 8, 8))
        layers['pool3_7'] = MaxPoolingLayer(2, 2)
        layers['dropout3_8'] = DropOut(0.5)

        layers['flatten'] = FlattenLayer((128, 4, 4), (2048, ))
        layers['fc4_1'] = FullyConnectedLayer(2048, 128, 0.001)
        layers['relu4_2'] = ReLULayer()
        layers['bn4_3'] = BatchNormLayer((128, ))
        layers['dropout4_4'] = DropOut(0.5)
        layers['fc4_5'] = FullyConnectedLayer(128, 10, 0.001)

        layers['softmax'] = SoftmaxLossLayer()
        model = {}
        model['layer_name'] = param_layer_name
        model['layers'] = layers

        return model