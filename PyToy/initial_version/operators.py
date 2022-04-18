# coding=utf-8
# import numpy as np
import cupy as cp
import numpy as np
import struct
import os
import time
from optimizer import AdamOptimizer, init_optimizer
from numba import njit, jit

class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output, std=0.01):  # 全连接层初始化
        self.num_input = num_input
        self.num_output = num_output
        self.std = std
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))

    def init_param(self, lr, optimizer):  # 参数初始化
        self.weight = cp.random.normal(loc=0.0, scale=self.std, size=(self.num_input, self.num_output))
        self.bias = cp.zeros([1, self.num_output])
        self.lr = lr
        self.optimizer_w = init_optimizer(lr, optimizer)
        self.optimizer_b = init_optimizer(lr, optimizer)

    def forward(self, input: cp.array, train:bool=True) -> cp.array:  # 前向传播计算
        self.input: cp.array = input
        output: cp.array = cp.add(cp.matmul(input, self.weight), self.bias)
        return output

    def backward(self, top_diff: cp.array) -> cp.array:  # 反向传播的计算
        self.d_weight: cp.array = cp.matmul(self.input.T, top_diff)
        self.d_bias: cp.array = top_diff.T
        bottom_diff: cp.array = cp.matmul(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self) -> None:  # 参数更新
        if not self.optimizer_w:
            self.weight: cp.array = self.weight - self.lr * self.d_weight
            self.bias: cp.array = self.bias - self.lr * cp.sum(self.d_bias, axis=1)
        else:
            self.weight: cp.array = self.optimizer_w.update(self.weight, self.d_weight)
            self.bias: cp.array = self.optimizer_b.update(self.bias, cp.sum(self.d_bias, axis=1))

    def load_param(self, param):  # 参数加载
        weight, bias = param
        assert self.weight.shape == weight.shape, "{} {}".format(self.weight.shape, weight.shape)
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def get_param(self):  # 参数保存
        return self.weight, self.bias

class ReLULayer(object):
    def __init__(self):
        print('\tReLU layer.')

    def forward(self, input, train=True):  # 前向传播的计算
        self.input = input
        return cp.maximum(self.input, 0)

    def backward(self, top_diff):  # 反向传播的计算
        return cp.where(self.input > 0, top_diff, 0)

class SoftmaxLossLayer(object):
    def __init__(self):
        self.eps = 1e-9
        print('\tSoftmax loss layer.')

    def forward(self, input, train=True):  # 前向传播的计算
        input_max = cp.max(input, axis=1, keepdims=True)
        input_exp = cp.exp(cp.subtract(input, input_max))
        self.prob = input_exp / cp.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):   # 计算损失
        self.batch_size = self.prob.shape[0]
        self.label_onehot = cp.zeros_like(self.prob)
        self.label_onehot[cp.arange(self.batch_size), label] = 1.0
        loss = -cp.sum(cp.multiply(cp.log(cp.add(self.prob, self.eps)), self.label_onehot)) / self.batch_size
        return loss

    def backward(self, top_diff):  # 反向传播的计算
        # top_diff here is useless, because we are starting from here
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, std=0.1):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.std = std
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))

    def init_param(self, lr, optimizer):
        self.weight = cp.random.normal(loc=0.0, scale=self.std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = cp.zeros([self.channel_out])
        self.lr = lr
        self.optimizer_w = init_optimizer(lr, optimizer)
        self.optimizer_b = init_optimizer(lr, optimizer)

    def forward(self, input, train=True):
        # [N, C, H, W]
        self.input_shape = input.shape
        height = input.shape[2] + 2 * self.padding
        width = input.shape[3] + 2 * self.padding
        # self.input_pad = cp.zeros([input.shape[0], input.shape[1], height, width])
        # self.input_pad[:, :, self.padding: self.padding + input.shape[2], self.padding: self.padding + input.shape[3]] = input
        height_out = int((height - self.kernel_size) / self.stride) + 1
        width_out = int((width - self.kernel_size) / self.stride) + 1
        mat_w = self.kernel_size * self.kernel_size * self.channel_in
        mat_h = height_out * width_out
        self.input_pad = cp.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')

        self.col = cp.empty((input.shape[0], mat_h, mat_w))
        cur = 0
        for x in range(height_out):
            for y in range(width_out):
                bias_x = x * self.stride
                bias_y = y * self.stride
                self.col[:, cur, :] = self.input_pad[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(input.shape[0], -1)
                cur = cur + 1
        output = cp.add(cp.matmul(self.col, self.weight.reshape(-1, self.weight.shape[-1])), self.bias)
        output = cp.moveaxis(output.reshape(input.shape[0], height_out, width_out, self.channel_out), 3, 1)

        return output

    def backward(self, top_diff):
        # top_diff batch, cout, h, w

        height_out = int((self.input_shape[2] + 2 * self.padding - self.kernel_size) / self.stride) + 1
        width_out = int((self.input_shape[3] + 2 * self.padding - self.kernel_size) / self.stride) + 1

        # cout, batch, h, w
        top_diff_col = cp.transpose(top_diff, [1, 0, 2, 3]).reshape(top_diff.shape[1], -1)
        # self.col batch, (h * w), (cin * k * k)

        # what we want, cin, k, k, cout
        tmp = cp.transpose(self.col.reshape(-1, self.col.shape[-1]), [1, 0])
        self.d_weight = cp.matmul(tmp, top_diff_col.T).reshape(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out)
        self.d_bias = top_diff_col.sum(axis=1)
        
        backward_col = cp.empty((top_diff.shape[0], self.input_shape[2] * self.input_shape[3], self.kernel_size * self.kernel_size * self.channel_out))
        pad_height = int(((self.input_shape[2] - 1) * self.stride + self.kernel_size - height_out) / 2)
        pad_width = int(((self.input_shape[3] - 1) * self.stride + self.kernel_size - width_out) / 2)
        # top_diff_pad = cp.zeros((top_diff.shape[0], top_diff.shape[1], height_out + 2 * pad_height, width_out + 2 * pad_width))
        # top_diff_pad[:, :, pad_height: height_out + pad_height, pad_width: width_out + pad_width] = top_diff
        # pad_height = (self.input_shape[2] - top_diff.shape[2]) // 2
        # pad_width = (self.input_shape[3] - top_diff.shape[3]) // 2
        top_diff_pad = cp.pad(top_diff, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)), 'constant')

        cur = 0
        for x in range(self.input_shape[2]):
            for y in range(self.input_shape[3]):
                bias_x = x * self.stride
                bias_y = y * self.stride
                backward_col[:, cur, :] = top_diff_pad[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(top_diff.shape[0], -1)
                cur = cur + 1

        # backward_col [batch, height * width, cout * k * k]
        # try to draw a draft and you will know the reason.
        # you shall consider the contribution from top_diff to the original dx
        # if x * kernel[i] has contribution to y, then dy * kernel[size - i] will have contribution
        weight_tmp = cp.transpose(self.weight, [3, 1, 2, 0]).reshape(self.channel_out, -1, self.channel_in)[:, ::-1, :].reshape(-1, self.channel_in)
        bottom_diff = cp.matmul(backward_col, weight_tmp)
        # [batch, height, width, cin]
        bottom_diff = cp.transpose(bottom_diff.reshape(top_diff.shape[0], self.input_shape[2], self.input_shape[3], self.input_shape[1]), [0, 3, 1, 2])
        
        return bottom_diff

    def get_gradient(self):
        return self.d_weight, self.d_bias

    def update_param(self):
        if not self.optimizer_w:
            self.weight += - self.lr * self.d_weight
            self.bias += - self.lr * self.d_bias
        else:
            self.weight = self.optimizer_w.update(self.weight, self.d_weight)
            self.bias = self.optimizer_b.update(self.bias, self.d_bias)

    def load_param(self, param):
        weight, bias = param
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def get_param(self):  # 参数保存
        return self.weight, self.bias

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))

    def forward(self, input, train=True):
        self.input_shape = input.shape # [N, C, H, W]
        height_out = int((self.input_shape[2] - self.kernel_size) / self.stride) + 1
        width_out = int((self.input_shape[3] - self.kernel_size) / self.stride) + 1
        mat_w = self.kernel_size * self.kernel_size
        mat_h = height_out * width_out

        col = cp.empty((input.shape[0], input.shape[1], mat_h, mat_w))
        cur = 0
        for x in range(height_out):
            for y in range(width_out):
                bias_x = x * self.stride
                bias_y = y * self.stride
                col[:, :, cur, :] = input[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(input.shape[0], input.shape[1], -1)
                cur = cur + 1

        output = cp.max(col, axis=3, keepdims=True)
        max_index = cp.argmax(col.reshape(input.shape[0], input.shape[1], height_out, width_out, self.kernel_size * self.kernel_size), axis=4)
        self.max_elements = cp.zeros((input.shape[0], input.shape[1], height_out, width_out, self.kernel_size * self.kernel_size))
        # https://stackoverflow.com/questions/44143438/numpy-indexing-set-1-to-max-value-and-zeros-to-all-others
        # refering the advanced indexing in numpy
        n, c, h, w = self.max_elements.shape[: 4]
        N, C, H, W = cp.ogrid[:n, :c, :h, :w]
        self.max_elements[N, C, H, W, max_index] = 1
        output = output.reshape(input.shape[0], input.shape[1], height_out, width_out)
        return output

    def backward(self, top_diff):
        bottom_diff = cp.zeros(self.input_shape)
        contrib = cp.multiply(self.max_elements, (top_diff.reshape(list(top_diff.shape) + [1])))
        for x in range(top_diff.shape[2]):
            for y in range(top_diff.shape[3]):
                bias_x = x * self.stride
                bias_y = y * self.stride
                bottom_diff[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size] = \
                    cp.add(contrib[:, :, x, y, :].reshape(top_diff.shape[0], top_diff.shape[1], self.kernel_size, self.kernel_size), \
                            bottom_diff[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size])
        return bottom_diff

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))

    def forward(self, input, train=True):
        assert list(input.shape[1:]) == list(self.input_shape)
        self.output = input.reshape([input.shape[0]] + list(self.output_shape))
        return self.output

    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))
        return bottom_diff

# referring https://github.com/renan-cunha/BatchNormalization
class BatchNormLayer(object):

    def __init__(self, dims: tuple) -> None:
        self.dims = dims

    def init_param(self, lr, optimizer):
        self.lr = lr
        self.optimizer_w = init_optimizer(lr, optimizer)
        self.optimizer_b = init_optimizer(lr, optimizer)

        self.gamma = cp.ones(([1] + list(self.dims)), dtype="float32")
        self.bias = cp.zeros(([1] + list(self.dims)), dtype="float32")

        self.running_mean_x = cp.zeros(0)
        self.running_var_x = cp.zeros(0)
        self.initialized = False

        # forward params
        self.var_x = cp.zeros(0)
        self.stddev_x = cp.zeros(0)
        self.x_minus_mean = cp.zeros(0)
        self.standard_x = cp.zeros(0)
        self.num_examples = 0
        self.mean_x = cp.zeros(0)
        self.running_avg_gamma = 0.9

        # backward params
        self.gamma_grad = cp.zeros(0)
        self.bias_grad = cp.zeros(0)

    def update_running_variables(self) -> None:
        # is_mean_empty = cp.array_equal(cp.zeros(0), self.running_mean_x)
        # is_var_empty = cp.array_equal(cp.zeros(0), self.running_var_x)
        # if is_mean_empty != is_var_empty:
        #     raise ValueError("Mean and Var running averages should be "
        #                      "initilizaded at the same time")
        if self.initialized:
            gamma = self.running_avg_gamma
            self.running_mean_x = gamma * self.running_mean_x + \
                                  (1.0 - gamma) * self.mean_x
            self.running_var_x = gamma * self.running_var_x + \
                                 (1. - gamma) * self.var_x
        else:
            self.running_mean_x = self.mean_x
            self.running_var_x = self.var_x
            self.initialized = True

    def forward(self, x: cp.ndarray, train: bool = True) -> cp.ndarray:
        self.num_examples = x.shape[0]
        if train:
            self.mean_x = cp.mean(x, axis=0, keepdims=True)
            # self.var_x = cp.mean(cp.power(cp.subtract(x, self.mean_x), 2), axis=0, keepdims=True)
            self.var_x = cp.var(x, axis=0, keepdims=True)
            self.update_running_variables()
        else:
            self.mean_x = self.running_mean_x.copy()
            self.var_x = self.running_var_x.copy()

        self.var_x += 1e-9
        self.stddev_x = cp.sqrt(self.var_x)
        self.x_minus_mean = cp.subtract(x, self.mean_x)
        self.standard_x = cp.divide(self.x_minus_mean, self.stddev_x)
        return self.gamma * self.standard_x + self.bias

    def backward(self, grad_input: cp.ndarray) -> cp.ndarray:
        standard_grad = grad_input * self.gamma

        var_grad = cp.sum(standard_grad * self.x_minus_mean * -0.5 * self.var_x ** (-3/2),
                          axis=0, keepdims=True)
        stddev_inv = 1 / self.stddev_x
        aux_x_minus_mean = 2 * self.x_minus_mean / self.num_examples

        mean_grad = (cp.sum(standard_grad * -stddev_inv, axis=0,
                            keepdims=True) +
                            var_grad * cp.sum(-aux_x_minus_mean, axis=0,
                            keepdims=True))

        self.gamma_grad = cp.sum(grad_input * self.standard_x, axis=0,
                                 keepdims=True)
        self.bias_grad = cp.sum(grad_input, axis=0, keepdims=True)

        return standard_grad * stddev_inv + var_grad * aux_x_minus_mean + \
               mean_grad / self.num_examples

    def update_param(self) -> None:
        if not self.optimizer_w:
            self.gamma -= self.lr * self.gamma_grad
            self.bias -= self.lr * self.bias_grad
        else:
            self.gamma = self.optimizer_w.update(self.gamma, self.gamma_grad)
            self.bias = self.optimizer_b.update(self.bias, self.bias_grad)

    def get_param(self):  # 参数保存
        return self.gamma, self.bias, self.running_mean_x, self.running_var_x, self.initialized

    def load_param(self, param):
        gamma, bias, running_mean_x, running_var_x, initialized = param
        self.gamma = gamma
        self.bias = bias
        self.running_mean_x = running_mean_x
        self.running_var_x = running_var_x
        self.initialized = initialized

class DropOut(object):
    def __init__(self, drop_prob) -> None:
        assert 0 < drop_prob < 1
        self.keep_prob = 1 - drop_prob
    
    def forward(self, input, train=True):
        if train:
            self.activated = cp.random.uniform(size=(input.shape)) < self.keep_prob
            return cp.multiply(self.activated, input) / self.keep_prob
        else:
            return input

    def backward(self, top_diff):
        return cp.multiply(top_diff, self.activated) / self.keep_prob
    