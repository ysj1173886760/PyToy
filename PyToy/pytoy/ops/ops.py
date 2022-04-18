# -*- encoding: utf-8 -*-
'''
@File    :   ops.py
@Time    :   2021/11/29 21:06:48
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   Operators
'''

import numpy as np
import numpy as np

from ..core import Node

# abc for operator
class Operator(Node):
    pass

class AddOperator(Operator):

    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.dims = parents[0].dims
        for parent in parents:
            if parent.dims > self.dims:
                self.dims = parent.dims

    def compute(self):
        self.value = np.zeros(self.parents[0].shape())
        for parent in self.parents:
            self.value = np.add(self.value, parent.value)

    def get_graident(self, parent):
        # also support dynamic boardcast here
        # but not support automatic squeeze the dims
        if self.graident.shape == parent.dims:
            return self.graident
        else:
            # calc the boardcast dims
            boardcast_dims = tuple([index for index, (i, j) in enumerate(zip(self.graident.shape, parent.dims)) if i != j])
            # reduce them
            return np.sum(self.graident, axis=boardcast_dims)

class MatMulOperator(Operator):
    
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        assert (parents[0].dims[1] == parents[1].dims[0])
        self.dims = (parents[0].dims[0], parents[1].dims[1])

    def compute(self):
        self.lhs = self.parents[0].value
        self.rhs = self.parents[1].value
        self.value = np.matmul(self.lhs, self.rhs)

    def get_graident(self, parent):
        # (m, n) = (m, p) * (p, n)
        if parent is self.parents[0]:
            return np.matmul(self.graident, self.rhs.T)
        else:
            return np.matmul(self.lhs.T, self.graident)

    def vacuum(self, forward=True):
        if forward:
            self.value = None
        else:
            self.graident = None
            self.lhs = None
            self.rhs = None

class DotOperator(Operator):
    
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        assert (parents[0].dims == parents[1].dims)
        self.dims = parents[0].dims

    def compute(self):
        self.lhs = self.parents[0].value
        self.rhs = self.parents[1].value
        self.value = np.multiply(self.lhs, self.rhs)

    def get_graident(self, parent):
        if parent is self.parents[0]:
            return np.multiply(self.graident, self.rhs)
        else:
            return np.multiply(self.graident, self.lhs)

class BoardcastOperator(Operator):
    
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.from_shape = parents[0].dims
        self.to_shape = kargs.get('to_shape')
        self.dims = self.to_shape

        assert len(self.from_shape) == len(self.to_shape)

        self.boardcast_dims = tuple([index for index, (i, j) in enumerate(zip(self.from_shape, self.to_shape)) if i != j])

    def compute(self):
        self.value = np.broadcast_to(self.parents[0].value, self.to_shape)
    
    def get_graident(self, parent):
        return np.sum(self.graident, axis=self.boardcast_dims, keepdims=True)
    
class ReshapeOperator(Operator):
    
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.from_shape = parents[0].dims
        self.to_shape = kargs.get('to_shape')
        self.dims = self.to_shape

        assert np.product(self.from_shape) == np.product(self.to_shape)
    
    def compute(self):
        self.value = np.reshape(self.parents[0].value, self.to_shape)
    
    def get_graident(self, parent):
        return np.reshape(self.graident, self.from_shape)

class SoftMax(Operator):
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)

    def compute(self):
        input_max = np.max(self.parents[0].value, axis=1)
        input_exp = np.exp(np.subtract(self.parents[0].value, input_max))
        self.value = input_exp / np.sum(input_exp, axis=1)
    
    def get_graident(self, parent):
        raise NotImplementedError("Do not use softmax to do BP")

class SigmoidOperator(Operator):
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.dims = parents[0].dims

    def compute(self):
        self.value = 1 / (1 + np.exp(-self.parents[0].value))
    
    def get_graident(self, parent):
        return np.multiply(self.value, (1-self.value))

class TanhOperator(Operator):
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.dims = parents[0].dims
    
    def compute(self):
        self.value = np.subtract(np.exp(self.parents[0].value), np.exp(-self.parents[0].value)) \
                    / np.add(np.exp(self.parents[0].value), np.exp(-self.parents[0].value))
    
    def get_graident(self, parent):
        return 1 - np.power(self.value, 2)

class ReLUOperator(Operator):
    
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.dims = parents[0].dims
        
    def compute(self):
        self.mask = self.parents[0].value > 0
        self.value = np.multiply(self.parents[0].value, self.mask)
    
    def get_graident(self, parent):
        return np.multiply(self.mask, self.graident)

    def vacuum(self, forward=True):
        if forward:
            self.value = None
        else:
            self.graident = None
            self.mask = None

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
        """[summary]
        params should have kernel_size, stride, padding, channel_in, channel_out
        """
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
        input_pad = np.pad(self.parents[0].value, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        self.col = np.empty((self.input_shape[0], self.mat_h, self.mat_w))
        cur = 0
        for x in range(self.height_out):
            for y in range(self.width_out):
                bias_x = x * self.stride
                bias_y = y * self.stride
                self.col[:, cur, :] = input_pad[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(self.input_shape[0], -1)
                cur = cur + 1
        self.value = np.matmul(self.col, self.parents[1].value.reshape(-1, self.parents[1].value.shape[-1]))
        self.value = np.moveaxis(self.value.reshape(self.input_shape[0], self.height_out, self.width_out, self.channel_out), 3, 1)
        self.weight = self.parents[1].value

    def get_graident(self, parent):
        if parent is self.parents[0]:
            # self.graident N, Cout, H, W
            # self.weight Cin, k, k, Cout
            # top_diff Cin, k, k, N, H, W
            top_diff = np.matmul(self.weight.reshape(-1, self.channel_out), 
                                    np.transpose(self.graident, [1, 0, 2, 3]).reshape(self.channel_out, -1)).reshape(self.channel_in * self.kernel_size * self.kernel_size, self.input_shape[0], -1)
            top_diff = np.transpose(top_diff, [1, 0, 2]).reshape(self.input_shape[0], self.channel_in * self.kernel_size * self.kernel_size, self.dims[2], self.dims[3])
            bottom_diff = np.empty((self.input_shape[0], self.channel_in, self.height, self.width))

            for x in range(self.dims[2]):
                for y in range(self.dims[3]):
                    bias_x = x * self.stride
                    bias_y = y * self.stride
                    bottom_diff[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size] = \
                        top_diff[:, :, x, y].reshape(self.graident.shape[0], self.channel_in, self.kernel_size, self.kernel_size)

            bottom_diff = bottom_diff[:, :, self.padding: self.padding + self.input_shape[2], self.padding: self.padding + self.input_shape[3]]
            return bottom_diff
        else:
            # same as above
            top_diff_col = np.transpose(self.graident, [1, 0, 2, 3]).reshape(self.graident.shape[1], -1)
            col_reshape = np.transpose(self.col.reshape(-1, self.col.shape[-1]), [1, 0])
            return np.matmul(col_reshape, top_diff_col.T).reshape(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out)

    def vacuum(self, forward=True):
        if forward:
            self.value = None
        else:
            self.graident = None
            self.mask = None
            self.col = None
            self.weight = None
    
    def tmp(self, parent):
        # TODO: figure out why this was wrong
        if parent is self.parents[0]:
            backward_col = np.empty((self.graident.shape[0], self.input_shape[2] * self.input_shape[3], self.kernel_size * self.kernel_size * self.channel_out))
            pad_height = int(((self.input_shape[2] - 1) * self.stride + self.kernel_size - self.height_out) / 2)
            pad_width = int(((self.input_shape[3] - 1) * self.stride + self.kernel_size - self.width_out) / 2)
            top_diff_pad = np.pad(self.graident, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)), 'constant')

            cur = 0
            for x in range(self.input_shape[2]):
                for y in range(self.input_shape[3]):
                    bias_x = x * self.stride
                    bias_y = y * self.stride
                    # cout * k * k
                    backward_col[:, cur, :] = top_diff_pad[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(self.graident.shape[0], -1)
                    cur = cur + 1

            weight_tmp = np.transpose(self.weight, [3, 1, 2, 0]).reshape(self.channel_out, -1, self.channel_in)[:, ::-1, :].reshape(-1, self.channel_in)
            bottom_diff = np.matmul(backward_col, weight_tmp)
            bottom_diff = np.transpose(bottom_diff.reshape(self.graident.shape[0], self.input_shape[2], self.input_shape[3], self.input_shape[1]), [0, 3, 1, 2])
            return bottom_diff
        else:
            # top_diff_col = np.transpose(top_diff, [1, 0, 2, 3]).reshape(top_diff.shape[1], -1)
            # self.d_weight = np.matmul(tmp, top_diff_col.T).reshape(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out)
            top_diff_col = np.transpose(self.graident, [1, 0, 2, 3]).reshape(self.graident.shape[1], -1)
            col_reshape = np.transpose(self.col.reshape(-1, self.col.shape[-1]), [1, 0])
            return np.matmul(col_reshape, top_diff_col.T).reshape(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out)
    
    def raw(self):
        bottom_diff = np.zeros(self.input_pad.shape)
        for idxn in range(self.graident.shape[0]):
            for idxc in range(self.graident.shape[1]):
                for idxh in range(self.graident.shape[2]):
                    for idxw in range(self.graident.shape[3]):
                        # TODO： 计算卷积层的反向传播， 权重、偏置的梯度和本层损失
                        bias_x = idxh * self.stride
                        bias_y = idxw * self.stride
                        bottom_diff[idxn, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size] += self.graident[idxn, idxc, idxh, idxw] * self.parents[1].value[:, :, :, idxc]
        bottom_diff = bottom_diff[:, :, self.padding: self.input_pad.shape[2] - self.padding, self.padding: self.input_pad.shape[3] - self.padding]
        return bottom_diff
            
def computeMse(input1, input2):
    return np.sum(np.square(input1.flatten() - input2.flatten()))

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
        col = np.empty((self.input_shape[0], self.input_shape[1], self.mat_h, self.mat_w))
        cur = 0
        for x in range(self.height_out):
            for y in range(self.width_out):
                bias_x = x * self.stride
                bias_y = y * self.stride
                col[:, :, cur, :] = self.parents[0].value[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(self.input_shape[0], self.input_shape[1], -1)
                cur = cur + 1

        output = np.max(col, axis=3, keepdims=True)
        max_index = np.argmax(col.reshape(self.input_shape[0], self.input_shape[1], self.height_out, self.width_out, self.kernel_size * self.kernel_size), axis=4)
        self.max_elements = np.zeros((self.input_shape[0], self.input_shape[1], self.height_out, self.width_out, self.kernel_size * self.kernel_size))
        n, c, h, w = self.max_elements.shape[: 4]
        N, C, H, W = np.ogrid[:n, :c, :h, :w]
        self.max_elements[N, C, H, W, max_index] = 1
        self.value = output.reshape(self.input_shape[0], self.input_shape[1], self.height_out, self.width_out)

    def get_graident(self, parents):
        bottom_diff = np.zeros(self.input_shape)
        contrib = np.multiply(self.max_elements, (self.graident.reshape(list(self.graident.shape) + [1])))
        for x in range(self.graident.shape[2]):
            for y in range(self.graident.shape[3]):
                bias_x = x * self.stride
                bias_y = y * self.stride
                bottom_diff[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size] = \
                    np.add(contrib[:, :, x, y, :].reshape(self.graident.shape[0], self.graident.shape[1], self.kernel_size, self.kernel_size), \
                            bottom_diff[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size])
        return bottom_diff

    def vacuum(self, forward=True):
        if forward:
            self.value = None
        else:
            self.graident = None
            self.max_elements = None

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
        self.gamma_grad = np.zeros(0)
        self.bias_grad = np.zeros(0)

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
            self.mean_x = np.mean(self.parents[0].value, axis=0, keepdims=True)
            self.var_x = np.var(self.parents[0].value, axis=0, keepdims=True)
            self.update_running_variables()
        else:
            if self.running_mean_x is not None:
                self.mean_x = self.running_mean_x
                self.var_x = self.running_var_x
            else:
                self.mean_x = 0.0
                self.var_x = 1.0

        self.var_x += 1e-9
        self.stddev_x = np.sqrt(self.var_x)
        self.x_minus_mean = np.subtract(self.parents[0].value, self.mean_x)
        self.standard_x = np.divide(self.x_minus_mean, self.stddev_x)
        self.value = self.parents[1].value * self.standard_x + self.parents[2].value
        self.rhs = self.parents[1].value

    def get_graident(self, parent):
        if parent == self.parents[0]:
            standard_grad = self.graident * self.rhs

            var_grad = np.sum(standard_grad * self.x_minus_mean * -0.5 * self.var_x ** (-3/2),
                            axis=0, keepdims=True)
            stddev_inv = 1 / self.stddev_x
            aux_x_minus_mean = 2 * self.x_minus_mean / self.batch_size

            mean_grad = (np.sum(standard_grad * -stddev_inv, axis=0,
                                keepdims=True) +
                                var_grad * np.sum(-aux_x_minus_mean, axis=0,
                                keepdims=True))
            return standard_grad * stddev_inv + var_grad * aux_x_minus_mean + \
                mean_grad / self.batch_size

        elif parent == self.parents[1]:
            return np.sum(self.graident * self.standard_x, axis=0, keepdims=True)

        else:
            return np.sum(self.graident, axis=0, keepdims=True)

    def vacuum(self, forward=True):
        if forward:
            self.value = None
        else:
            self.graident = None
            self.x_minus_mean = None
            self.standard_x = None
            self.rhs = None
            self.var_x = None
            self.stddev_x = None

class DropOutOperator(Operator):
    
    # for the sake of simplicity saving, i saved all the params in kargs
    def __init__(self, *parents, **kargs) -> None:
        """[summary]
        drop_prob: drop probability
        """
        Operator.__init__(self, *parents, **kargs)
        self.dims = parents[0].dims
        drop_prob = kargs.get('drop_prob')
        assert 0 < drop_prob < 1
        self.keep_prob = 1 - drop_prob

    def compute(self):
        if self.graph.training:
            self.activated = np.random.uniform(size=(self.dims)) < self.keep_prob
            self.value = np.multiply(self.activated, self.parents[0].value) / self.keep_prob
        else:
            self.value = self.parents[0].value

    def get_graident(self, parent):
        return np.multiply(self.graident, self.activated) / self.keep_prob

    def vacuum(self, forward=True):
        if forward:
            self.value = None
        else:
            self.graident = None
            self.activated = None

class AvgPoolingOperator(Operator):
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
        col = np.empty((self.input_shape[0], self.input_shape[1], self.mat_h, self.mat_w))
        cur = 0
        for x in range(self.height_out):
            for y in range(self.width_out):
                bias_x = x * self.stride
                bias_y = y * self.stride
                col[:, :, cur, :] = self.parents[0].value[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size].reshape(self.input_shape[0], self.input_shape[1], -1)
                cur = cur + 1

        output = np.mean(col, axis=3)
        self.value = output.reshape(self.input_shape[0], self.input_shape[1], self.height_out, self.width_out)

    def get_graident(self, parents):
        bottom_diff = np.zeros(self.input_shape)
        area = self.kernel_size * self.kernel_size
        for x in range(self.graident.shape[2]):
            for y in range(self.graident.shape[3]):
                bias_x = x * self.stride
                bias_y = y * self.stride
                bottom_diff[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size] = \
                    np.add(np.repeat(self.graident[:, :, x, y].reshape(self.input_shape[0], self.input_shape[1], 1), area, axis=2).reshape(self.input_shape[0], self.input_shape[1], self.kernel_size, self.kernel_size) / area, \
                            bottom_diff[:, :, bias_x: bias_x + self.kernel_size, bias_y: bias_y + self.kernel_size])
        return bottom_diff

class Welding(Operator):
    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.dims = parents[0].dims

    def compute(self, parents):
        assert len(self.parents) == 1 and self.parents[0] is not None
        self.value = self.parents[0].value
    
    def get_graident(self, parent):
        assert parent is self.parents[0]
        return self.graident

    def weld(self, node):
        if len(self.parents) == 1 and self.parents[0] is not None:
            self.parents[0].children.remove(self)
        self.parents.clear()

        self.parents.append(node)
        node.children.append(self)
