import numpy as np
from tqdm import tqdm_notebook

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