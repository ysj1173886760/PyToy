import os
import numpy as np

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