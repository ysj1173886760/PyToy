# -*- encoding: utf-8 -*-
'''
@File    :   loss.py
@Time    :   2021/11/29 21:33:53
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   Loss function
'''

import numpy as cp
from ..core import Node

# abc for loss function
class LossFunction(Node):
    pass

class L2Loss(LossFunction):
    # assume the param matrixes are [batch, features]
    def __init__(self, *parents, **kargs) -> None:
        LossFunction.__init__(self, *parents, **kargs)
        self.batch_size = parents[0].dims[0]

    def compute(self):
        self.value = cp.sum(cp.square(self.parents[0].value - self.parents[1].value)) / self.batch_size

    def get_graident(self, parent):
        
        if parent is self.parents[0]:
            return 2 * cp.subtract(self.parents[0].value, self.parents[1].value) / self.batch_size
        else:
            return 2 * cp.subtract(self.parents[1].value, self.parents[0].value) / self.batch_size

class CrossEntropyWithSoftMax(LossFunction):
    # assume the param matrixes are [batch, features]
    # parent[0] is input, parent[1] is label
    def __init__(self, *parents, **kargs) -> None:
        LossFunction.__init__(self, *parents, **kargs)
        self.batch_size = parents[0].dims[0]
        self.eps = 1e-9

    def compute(self):
        input_max = cp.max(self.parents[0].value, axis=1, keepdims=True)
        input_exp = cp.exp(cp.subtract(self.parents[0].value, input_max))
        self.prob = input_exp / cp.sum(input_exp, axis=1, keepdims=True)

        self.label_onehot = cp.zeros_like(self.prob)
        self.label_onehot[cp.arange(self.batch_size), self.parents[1].value] = 1.0
        loss = -cp.sum(cp.multiply(cp.log(cp.add(self.prob, self.eps)), self.label_onehot)) / self.batch_size
        return loss
    
    def get_graident(self, parent):
        return (self.prob - self.label_onehot) / self.batch_size