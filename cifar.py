import cupy as cp
import numpy as np
import time
import pytoy as pt
import os

from pytoy.core.node import Variable
from pytoy.layer.layer import Conv, Dense, Flatten, MaxPooling
from pytoy.ops.loss import CrossEntropyWithSoftMax
from pytoy.ops.ops import ReLU, SoftMax
import tqdm

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(data_dir, data_list):
    for i, pth in enumerate(data_list):
        data_pth = os.path.join(data_dir, pth)
        data = unpickle(data_pth)
        images = data[b'data'].reshape(-1, 3, 32, 32)
        labels = np.array(data[b'labels']).reshape(-1)
        if i == 0:
            res_data = images
            res_label = labels
        else:
            res_data = np.concatenate((res_data, images))
            res_label = np.concatenate((res_label, labels))
    return res_data, res_label

class dataAugumentor():
    def __init__(self, toTensor=True, whiten=True, crop=True, rotate=True, flip=True, noise=True) -> None:
        self.toTensor = toTensor
        self.whiten = whiten
        self.crop = crop
        self.rotate= rotate
        self.flip = flip
        self.noise = noise

    def calc_mean_std(self, data, axis):
        mean = np.mean(data, axis=axis)
        std = np.var(data, axis=axis)
        return mean, std
    
    def img_whiten(self, data):
        for i in range(3):
            data[:, i, :, :] = (data[:, i, :, :] - self.mean[i]) / self.std[i]
        return data
    
    def img_crop(self, data):
        for idxn in range(data.shape[0]):
            pad_image = np.pad(data[idxn], [[0, 0], [4, 4], [4, 4]], 'constant')
            left = np.random.randint(0, pad_image.shape[1] - data[idxn].shape[1] + 1)
            top = np.random.randint(0, pad_image.shape[2] - data[idxn].shape[2] + 1)
            data[idxn] = pad_image[:, left: left + data[idxn].shape[1], top: top + data[idxn].shape[2]]
        return data

    def img_rotate(self, data):
        for idxn in range(data.shape[0]):
            k = np.random.randint(0, 4)
            data[idxn] = np.rot90(data[idxn], k, axes=(1, 2))
        return data
    
    def img_flip(self, data):
        for idxn in range(data.shape[0]):
            rand = np.random.random()
            if rand < 0.3:
                data[idxn] = np.flip(data[idxn], axis=2)
            elif rand < 0.6:
                data[idxn] = np.flip(data[idxn], axis=1)
        return data

    def augument(self, data, train_data=True):
        if train_data:
            if self.toTensor:
                data = np.array(data / 255, dtype=np.float32)
                self.mean, self.std = self.calc_mean_std(data, (0, 2, 3))
            if self.rotate:
                data = self.img_rotate(data)
            if self.crop:
                data = self.img_crop(data)
            if self.flip:
                data = self.img_flip(data)
            if self.whiten:
                data = self.img_whiten(data)
        else:
            if self.toTensor:
                data = np.array(data / 255, dtype=np.float32)
            if self.whiten:
                data = self.img_whiten(data)

        return data

class CIFAR(object):
    def build(self):
        self.input = Variable((BATCH_SIZE, 3, 32, 32), init=False, trainable=False)
        self.label = Variable((BATCH_SIZE, ), init=False, trainable=False)
        self.layers = {}
        self.layers['conv1_1'] = Conv(self.input, 3, 32, 3, 1, 1, name='conv1_1', std=0.1)
        self.layers['relu1_2'] = ReLU(self.layers['conv1_1'], prefix='relu1_2')
        self.layers['pool1'] = MaxPooling(self.layers['relu1_2'], 2, 2, name='pool1')

        self.layers['conv2_1'] = Conv(self.layers['pool1'], 32, 64, 3, 1, 1, name='conv2_1', std=0.1)
        self.layers['relu2_2'] = ReLU(self.layers['conv2_1'], prefix='relu2_2')
        self.layers['pool2'] = MaxPooling(self.layers['relu2_2'], 2, 2, name='pool2')

        self.layers['conv3_1'] = Conv(self.layers['pool2'], 64, 128, 3, 1, 1, name='conv3_1', std=0.01)
        self.layers['relu3_2'] = ReLU(self.layers['conv3_1'], prefix='relu3_2')
        self.layers['pool3'] = MaxPooling(self.layers['relu3_2'], 2, 2, name='pool3')

        self.layers['flatten'] = Flatten(self.layers['pool3'])
        self.layers['fc1'] = Dense(self.layers['flatten'], 2048, 10, std=0.01)

        self.layers['softmax'] = SoftMax(self.layers['fc1'])
        self.layers['loss'] = CrossEntropyWithSoftMax(self.layers['fc1'], self.label)
        
    
    def evaluate(self, test_data, test_label):
        test = cp.array(test_data)
        label = cp.array(test_label)
        pred_results = cp.zeros([test.shape[0]])
        total_loss = 0
        for idx in range(int(test.shape[0] / BATCH_SIZE)):
            batch_images = test[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
            self.input.set_value(batch_images)
            self.label.set_value(label[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE])
            self.layers['softmax'].forward()
            self.layers['loss'].forward()
            total_loss += self.layers['loss'].value
            pred_labels = cp.argmax(self.layers['softmax'].value, axis=1)
            pred_results[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE] = pred_labels
        accuracy = cp.mean(pred_results == label)
        return accuracy, total_loss

    def train(self, train_data, train_label, test_data, test_label):
        random_index = np.arange(train_data.shape[0]).astype(int)
        max_batch = train_data.shape[0] // BATCH_SIZE
        last_accuracy = 0.0
        train_accuracy = 0.0
        validation_loss = 0.0

        adam = pt.optimizer.Adam(pt.default_graph, self.layers['loss'], LEARNING_RATE)
        # tqdm stuff
        for epoch in range(100):
            np.random.shuffle(random_index)
            train_data = train_data[random_index]
            train_label = train_label[random_index]
            bar = tqdm.tqdm(range(max_batch))
            total_loss = 0
            for cur in bar:
                batch_image = cp.array(train_data[cur * BATCH_SIZE: (cur + 1) * BATCH_SIZE])
                batch_label = cp.array(train_label[cur * BATCH_SIZE: (cur + 1) * BATCH_SIZE])

                # time1 = time.time()
                self.input.set_value(batch_image)
                self.label.set_value(batch_label)
                # time2 = time.time()
                # self.layers['loss'].forward()
                # time3 = time.time()
                # total_loss += self.layers['loss'].value

                # time4 = time.time()
                # for node in trainable_node:
                #     node.backward(self.layers['loss'])

                # time5 = time.time()
                # for node in trainable_node:
                #     node.set_value(node.value - LEARNING_RATE * node.graident)
                # time6 = time.time()
                
                # pt.default_graph.clear_graident()
                # time7 = time.time()
                # print('%f %f %f %f %f %f' % (time2 - time1, time3 - time2, time4 - time3,
                #                              time5 - time4, time6 - time5, time7 - time6))
                adam.step()
                total_loss += self.layers['loss'].value
                adam.update()

                bar.set_description("Epoch %d Loss %.6f ValidationLoss %.3f ValidationAccuracy %.3f TrainAccuracy %.3f" % (epoch, total_loss / (cur + 1), validation_loss, last_accuracy, train_accuracy))
                
            last_accuracy, validation_loss = self.evaluate(test_data, test_label)

if __name__ == '__main__':
    LEARNING_RATE = 0.01
    DATA_DIR = './data'
    BATCH_SIZE = 100
    data_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_list = ['test_batch']
    
    # load train data
    train_data, train_label = load_data(DATA_DIR, data_list)
    # load test data
    test_data, test_label = load_data(DATA_DIR, test_list)

    augumentor = dataAugumentor(toTensor=True, whiten=True, crop=False, rotate=False, flip=False, noise=False)

    train_data = augumentor.augument(train_data, True)
    test_data = augumentor.augument(test_data, False)
    cifar = CIFAR()
    cifar.build()
    cifar.train(train_data, train_label, test_data, test_label)