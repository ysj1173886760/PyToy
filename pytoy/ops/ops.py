# -*- encoding: utf-8 -*-
'''
@File    :   ops.py
@Time    :   2021/11/29 21:06:48
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   Operators
'''

import cupy as cp
import numpy as np
from ..core import Node

# abc for operator
class Operator(Node):
    pass

class Add(Operator):

    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.dims = parents[0].dims
        for parent in parents:
            if parent.dims > self.dims:
                self.dims = parent.dims

    def compute(self):
        self.value = cp.zeros(self.parents[0].shape())
        for parent in self.parents:
            self.value = cp.add(self.value, parent.value)

    def get_graident(self, parent):
        # also support dynamic boardcast here
        # but not support automatic squeeze the dims
        if self.graident.shape == parent.value.shape:
            return self.graident
        else:
            # calc the boardcast dims
            boardcast_dims = tuple([index for index, (i, j) in enumerate(zip(self.graident.shape, parent.value.shape)) if i != j])
            # reduce them
            return cp.sum(self.graident, axis=boardcast_dims, keepdims=True)

class MatMul(Operator):
    
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        assert (parents[0].dims[1] == parents[1].dims[0])
        self.dims = (parents[0].dims[0], parents[1].dims[1])

    def compute(self):
        self.value = cp.matmul(self.parents[0].value, self.parents[1].value)

    def get_graident(self, parent):
        # (m, n) = (m, p) * (p, n)
        if parent is self.parents[0]:
            return cp.matmul(self.graident, self.parents[1].value.T)
        else:
            return cp.matmul(self.parents[0].value.T, self.graident)

class Boardcast(Operator):
    
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.from_shape = parents[0].dims
        self.to_shape = kargs.get('to_shape')
        self.dims = self.to_shape

        assert len(self.from_shape) == len(self.to_shape)

        self.boardcast_dims = tuple([index for index, (i, j) in enumerate(zip(self.from_shape, self.to_shape)) if i != j])

    def compute(self):
        self.value = cp.broadcast_to(self.parents[0].value, self.to_shape)
    
    def get_graident(self, parent):
        return cp.sum(self.graident, axis=self.boardcast_dims, keepdims=True)
    
class Reshape(Operator):
    
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.from_shape = parents[0].dims
        self.to_shape = kargs.get('to_shape')
        self.dims = self.to_shape

        assert np.product(self.from_shape) == np.product(self.to_shape)
    
    def compute(self):
        self.value = cp.reshape(self.parents[0].value, self.to_shape)
    
    def get_graident(self, parent):
        return cp.reshape(self.graident, self.from_shape)

class SoftMax(Operator):
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)

    def compute(self):
        input_max = cp.max(self.parents[0].value, axis=1, keepdims=True)
        input_exp = cp.exp(cp.subtract(self.parents[0].value, input_max))
        self.value = input_exp / cp.sum(input_exp, axis=1, keepdims=True)
    
    def get_graident(self, parent):
        raise NotImplementedError("Do not use softmax to do BP")

class ReLU(Operator):
    
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.dims = parents[0].dims
        
    def compute(self):
        self.value = cp.maximum(self.parents[0].value, 0)
    
    def get_graident(self, parent):
        return cp.where(self.value > 0, self.graident, 0)

class ConvOperator(Operator):
    # assume parent[0] is input, parent[1] is kernel
    # request input dims like [N, C, H, W]
    """[Conv]
    first input is feature [N, C, H, W],
    second input is kernel [Cin, K, K, Cout]

    Args:
        Operator ([type]): [description]
    """
    
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.kernel_size = kargs.get('kernel_size')
        self.stride = kargs.get('stride')
        self.padding = kargs.get('padding')
        self.channel_in = kargs.get('channel_in')
        self.channel_out = kargs.get('channel_out')

        # pre-compute
        self.input_shape = parents[0].dims
        self.height = self.input_shape[2] + 2 * self.padding
        self.width = self.input_shape[3] + 2 * self.padding
        self.height_out = int((self.height - self.kernel_size) / self.stride) + 1
        self.width_out = int((self.width - self.kernel_size) / self.stride) + 1
        self.mat_w = self.kernel_size * self.kernel_size * self.channel_in
        self.mat_h = self.height_out * self.width_out

        self.dims = (self.input_shape[0], self.channel_out, self.height_out, self.width_out)
    
    def compute(self):
        self.input_pad = cp.pad(self.parents[0].value, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        self.col = cp.empty((self.input_shape[0], self.mat_h, self.mat_w))
        cur = 0
        for x in range(self.height_out):
            for y in range(self.width_out):
                bias_x = x * self.stride
                bias_y = y * self.stride
                self.col[:, cur, :] = self.input_pad[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(self.input_shape[0], -1)
                cur = cur + 1
        self.value = cp.matmul(self.col, self.parents[1].value.reshape(-1, self.parents[1].value.shape[-1]))
        self.value = cp.moveaxis(self.value.reshape(self.input_shape[0], self.height_out, self.width_out, self.channel_out), 3, 1)

    # refer to initial version
    def get_graident(self, parent):
        if parent is self.parents[0]:
            backward_col = cp.empty((self.graident.shape[0], self.input_shape[2] * self.input_shape[3], self.kernel_size * self.kernel_size * self.channel_out))
            pad_height = int(((self.input_shape[2] - 1) * self.stride + self.kernel_size - self.height_out) / 2)
            pad_width = int(((self.input_shape[3] - 1) * self.stride + self.kernel_size - self.width_out) / 2)
            top_diff_pad = cp.pad(self.graident, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)), 'constant')

            cur = 0
            for x in range(self.input_shape[2]):
                for y in range(self.input_shape[3]):
                    bias_x = x * self.stride
                    bias_y = y * self.stride
                    backward_col[:, cur, :] = top_diff_pad[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(self.graident.shape[0], -1)
                    cur = cur + 1

            weight_tmp = cp.transpose(self.parents[1].value, [3, 1, 2, 0]).reshape(self.channel_out, -1, self.channel_in)[:, ::-1, :].reshape(-1, self.channel_in)
            bottom_diff = cp.matmul(backward_col, weight_tmp)
            bottom_diff = cp.transpose(bottom_diff.reshape(self.graident.shape[0], self.input_shape[2], self.input_shape[3], self.input_shape[1]), [0, 3, 1, 2])
            return bottom_diff
        else:
            # top_diff_col = cp.transpose(top_diff, [1, 0, 2, 3]).reshape(top_diff.shape[1], -1)
            # self.d_weight = cp.matmul(tmp, top_diff_col.T).reshape(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out)
            top_diff_col = cp.transpose(self.graident, [1, 0, 2, 3]).reshape(self.graident.shape[1], -1)
            col_reshape = cp.transpose(self.col.reshape(-1, self.col.shape[-1]), [1, 0])
            return cp.matmul(col_reshape, top_diff_col.T).reshape(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out)

class MaxPoolingOperator(Operator):
    """[summary]
    input dims are [N, C, H, W]

    Args:
        Operator ([type]): [description]
    """
    
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.kernel_size = kargs.get('kernel_size')
        self.stride = kargs.get('stride')

        self.input_shape = parents[0].dims # [N, C, H, W]
        self.height_out = int((self.input_shape[2] - self.kernel_size) / self.stride) + 1
        self.width_out = int((self.input_shape[3] - self.kernel_size) / self.stride) + 1
        self.mat_w = self.kernel_size * self.kernel_size
        self.mat_h = self.height_out * self.width_out
        self.dims = (self.input_shape[0], self.input_shape[1], self.height_out, self.width_out)

    def compute(self):
        col = cp.empty((self.input_shape[0], self.input_shape[1], self.mat_h, self.mat_w))
        cur = 0
        for x in range(self.height_out):
            for y in range(self.width_out):
                bias_x = x * self.stride
                bias_y = y * self.stride
                col[:, :, cur, :] = self.parents[0].value[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(self.input_shape[0], self.input_shape[1], -1)
                cur = cur + 1

        output = cp.max(col, axis=3, keepdims=True)
        max_index = cp.argmax(col.reshape(self.input_shape[0], self.input_shape[1], self.height_out, self.width_out, self.kernel_size * self.kernel_size), axis=4)
        self.max_elements = cp.zeros((self.input_shape[0], self.input_shape[1], self.height_out, self.width_out, self.kernel_size * self.kernel_size))
        n, c, h, w = self.max_elements.shape[: 4]
        N, C, H, W = cp.ogrid[:n, :c, :h, :w]
        self.max_elements[N, C, H, W, max_index] = 1
        self.value = output.reshape(self.input_shape[0], self.input_shape[1], self.height_out, self.width_out)

    def get_graident(self, parents):
        bottom_diff = cp.zeros(self.input_shape)
        contrib = cp.multiply(self.max_elements, (self.graident.reshape(list(self.graident.shape) + [1])))
        for x in range(self.graident.shape[2]):
            for y in range(self.graident.shape[3]):
                bias_x = x * self.stride
                bias_y = y * self.stride
                bottom_diff[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size] = \
                    cp.add(contrib[:, :, x, y, :].reshape(self.graident.shape[0], self.graident.shape[1], self.kernel_size, self.kernel_size), \
                            bottom_diff[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size])
        return bottom_diff

class BatchNormOperator(Operator):
    """[summary]
    we got 3 parents here.
    first is input, should be formatted as [batch_size, ...],
    second is gamma, should be formatted as [1, ...], where ... is same as above,
    third is bias, should be formatted as [1, ...], where ... is same as above

    Args:
        Operator ([type]): [description]
    """

    # TODO: BatchNorm can be rewritten to the combination of basic operators
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.dims = parents[0].dims

        self.running_mean_x = None
        self.running_var_x = None

        # forward params
        self.batch_size = self.dims[0]
        self.running_avg_gamma = 0.9

        # backward params
        self.gamma_grad = cp.zeros(0)
        self.bias_grad = cp.zeros(0)

    def update_running_variables(self) -> None:
        if self.running_mean_x is not None:
            gamma = self.running_avg_gamma
            self.running_mean_x = gamma * self.running_mean_x + \
                                  (1.0 - gamma) * self.mean_x
            self.running_var_x = gamma * self.running_var_x + \
                                 (1. - gamma) * self.var_x
        else:
            self.running_mean_x = self.mean_x
            self.running_var_x = self.var_x

    def compute(self):
        if self.graph.training:
            self.mean_x = cp.mean(self.parents[0].value, axis=0, keepdims=True)
            self.var_x = cp.var(self.parents[0].value, axis=0, keepdims=True)
            self.update_running_variables()
        else:
            if self.running_mean_x is not None:
                self.mean_x = self.running_mean_x.copy()
                self.var_x = self.running_var_x.copy()
            else:
                self.mean_x = 0.0
                self.var_x = 1.0

        self.var_x += 1e-9
        self.stddev_x = cp.sqrt(self.var_x)
        self.x_minus_mean = cp.subtract(self.parents[0].value, self.mean_x)
        self.standard_x = cp.divide(self.x_minus_mean, self.stddev_x)
        self.value = self.parents[1].value * self.standard_x + self.parents[2].value

    def get_graident(self, parent):
        if parent == self.parents[0]:
            standard_grad = self.graident * self.parents[1].value

            var_grad = cp.sum(standard_grad * self.x_minus_mean * -0.5 * self.var_x ** (-3/2),
                            axis=0, keepdims=True)
            stddev_inv = 1 / self.stddev_x
            aux_x_minus_mean = 2 * self.x_minus_mean / self.batch_size

            mean_grad = (cp.sum(standard_grad * -stddev_inv, axis=0,
                                keepdims=True) +
                                var_grad * cp.sum(-aux_x_minus_mean, axis=0,
                                keepdims=True))
            return standard_grad * stddev_inv + var_grad * aux_x_minus_mean + \
                mean_grad / self.batch_size

        elif parent == self.parents[1]:
            return cp.sum(self.graident * self.standard_x, axis=0, keepdims=True)

        else:
            return cp.sum(self.graident, axis=0, keepdims=True)