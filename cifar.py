import cupy as cp
import numpy as np
import time
import pytoy as pt
import os
from pytoy.core.core import get_node_from_graph

from pytoy.core.node import Variable
from pytoy.layer.layer import BatchNorm, Conv, Dense, Flatten, MaxPooling, DropOut, ReLU
from pytoy.ops.loss import CrossEntropyWithSoftMax
from pytoy.ops.ops import SoftMax
import tqdm
import cProfile

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
        self.input = Variable((BATCH_SIZE, 3, 32, 32), init=False, trainable=False, name='input')
        self.label = Variable((BATCH_SIZE, ), init=False, trainable=False, name='label')

        net = Conv(self.input, 3, 32, 3, 1, 1, name='conv1_1', std=0.001)
        net = ReLU(net, name='relu1_2')
        net = BatchNorm(net, name='bn1_3')
        net = Conv(net, 32, 32, 3, 1, 1, name='conv1_4', std=0.001)
        net = ReLU(net, name='relu1_5')
        net = BatchNorm(net, name='bn1_6')
        net = MaxPooling(net, 2, 2, name='pool1_7')
        net = DropOut(net, 0.3, name='dropout1_8')

        net = Conv(net, 32, 64, 3, 1, 1, name='conv2_1', std=0.001)
        net = ReLU(net, name='relu2_2')
        net = BatchNorm(net, name='bn2_3')
        net = Conv(net, 64, 64, 3, 1, 1, name='conv2_4', std=0.001)
        net = ReLU(net, name='relu2_5')
        net = BatchNorm(net, name='bn2_6')
        net = MaxPooling(net, 2, 2, name='pool2_7')
        net = DropOut(net, 0.5, name='dropout2_8')

        net = Conv(net, 64, 128, 3, 1, 1, name='conv3_1', std=0.001)
        net = ReLU(net, name='relu3_2')
        net = BatchNorm(net, name='bn3_3')
        net = Conv(net, 128, 128, 3, 1, 1, name='conv3_4', std=0.001)
        net = ReLU(net, name='relu3_5')
        net = BatchNorm(net, name='bn3_6')
        net = MaxPooling(net, 2, 2, name='pool3_7')
        net = DropOut(net, 0.5, name='dropout3_8')

        net = Flatten(net, name = 'flatten')
        net = Dense(net, 2048, 128, name='fc4_1', std=0.001)
        net = ReLU(net, name='relu4_2')
        net = BatchNorm(net, name='bn4_3')
        net = DropOut(net, 0.5, name='dropout4_4')
        net = Dense(net, 128, 10, name='fc4_5', std=0.001)

        # net = Conv(self.input, 3, 32, 3, 1, 1, name='conv1_1', std=0.001)
        # net = BatchNorm(net, name='bn1_2')
        # net = ReLU(net, name='relu1_3')
        # net = MaxPooling(net, 2, 2, name='pool1_4')

        # net = Conv(net, 32, 64, 3, 1, 1, name='conv2_1', std=0.001)
        # net = BatchNorm(net, name='bn2_2')
        # net = ReLU(net, name='relu2_3')
        # net = MaxPooling(net, 2, 2, name='pool2_4')

        # net = Conv(net, 64, 128, 3, 1, 1, name='conv3_1', std=0.001)
        # net = BatchNorm(net, name='bn3_2')
        # net = ReLU(net, name='relu3_3')
        # net = MaxPooling(net, 2, 2, name='pool3_4')

        # net = Flatten(net, name = 'flatten')
        # net = Dense(net, 2048, 10, name='fc4_1', std=0.001)

        self.softmax = SoftMax(net)
        self.loss = CrossEntropyWithSoftMax(net, self.label)
    
    def evaluate(self, test_data, test_label):
        test = cp.array(test_data)
        label = cp.array(test_label)
        pred_results = cp.zeros([test.shape[0]])
        total_loss = 0
        self.graph.evaluate()
        
        for idx in range(int(test.shape[0] / BATCH_SIZE)):
            batch_images = test[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
            self.input.set_value(batch_images)
            self.label.set_value(label[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE])
            self.softmax.forward()
            self.loss.forward()
            total_loss += self.loss.value
            pred_labels = cp.argmax(self.softmax.value, axis=1)
            pred_results[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE] = pred_labels
        accuracy = cp.mean(pred_results == label)
        return accuracy, total_loss

    def train(self, train_data, train_label, test_data, test_label):
        random_index = np.arange(train_data.shape[0]).astype(int)
        max_batch = train_data.shape[0] // BATCH_SIZE
        last_accuracy = 0.0
        train_accuracy = 0.0
        validation_loss = 0.0
        self.graph = pt.default_graph

        saver = pt.saver.Saver('./model')
        adam = pt.optimizer.Adam(pt.default_graph, self.loss, LEARNING_RATE)
        trainer = pt.trainer.Trainer(adam)
        # tqdm stuff
        for epoch in range(100):
            np.random.shuffle(random_index)
            train_data = train_data[random_index]
            train_label = train_label[random_index]
            bar = tqdm.tqdm(range(max_batch))
            total_loss = 0
            for cur in bar:
                self.graph.train()
                batch_image = cp.array(train_data[cur * BATCH_SIZE: (cur + 1) * BATCH_SIZE])
                batch_label = cp.array(train_label[cur * BATCH_SIZE: (cur + 1) * BATCH_SIZE])

                # self.input.set_value(batch_image)
                # self.label.set_value(batch_label)
                # adam.step()

                # prof = cProfile.Profile()
                # prof.enable()
                trainer.train({'input': batch_image, 'label': batch_label})
                # prof.create_stats()
                # prof.print_stats()

                total_loss += self.loss.value
                adam.update()

                bar.set_description("Epoch %d Loss %.6f ValidationLoss %.3f ValidationAccuracy %.3f TrainAccuracy %.3f" % (epoch, total_loss / (cur + 1), validation_loss, last_accuracy, train_accuracy))

            # saver.save()
            last_accuracy, validation_loss = self.evaluate(test_data, test_label)

def computeMse(input1, input2):
    return np.sum(np.square(input1.flatten() - input2.flatten()))

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