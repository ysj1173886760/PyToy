import numpy as np
from tqdm import tqdm_notebook

class dataAugumentor():
    def __init__(self, toTensor=True, whiten=True, crop=True, flip=True, noise=True) -> None:
        self.toTensor = toTensor
        self.whiten = whiten
        self.crop = crop
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
        
    def augument(self, data, train_data=True):
        if train_data:
            if self.toTensor:
                data = data / 255
            if self.whiten:
                self.mean, self.std = self.calc_mean_std(data, (0, 2, 3))
                data = self.img_whiten(data)
        else:
            if self.toTensor:
                data = data / 255
            if self.whiten:
                data = self.img_whiten(data)

        return data