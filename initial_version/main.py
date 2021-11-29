# coding:utf-8
import numpy as np
import cupy as cp
import os
import imageio
import time
from data_augumentation import dataAugumentor
from dataloader import load_data
from network import DeeperNetwork, Network, lightWeightNetwork, testNetwork
import cProfile

if __name__ == '__main__':
    LEARNING_RATE = 0.001
    DATA_DIR = './data'
    data_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_list = ['test_batch']

    net = Network(LEARNING_RATE, 'Adam')
    net.build_model(testNetwork().get_model())
    net.init_model()
    
    # load train data
    train_data, train_label = load_data(DATA_DIR, data_list)
    # load test data
    test_data, test_label = load_data(DATA_DIR, test_list)

    # preprocess data
    augumentor = dataAugumentor(toTensor=True, whiten=True, crop=True, rotate=False, flip=True, noise=False)

    train_data = augumentor.augument(train_data, True)
    test_data = augumentor.augument(test_data, False)
    net.train(train_data, train_label, test_data, test_label)
