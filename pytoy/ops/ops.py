# -*- encoding: utf-8 -*-
'''
@File    :   ops.py
@Time    :   2021/11/29 21:06:48
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   Operators
'''

import numpy as cp
from ..core import Node

# abc for operator
class Operator(Node):
    pass

class Add(Operator):

    def __init__(self, *parents, **kargs) -> None:
        Operator.__init__(self, *parents, **kargs)
        self.dims = parents[0].dims

    def compute(self):
        self.value = cp.zeros(self.parents[0].shape())
        for parent in self.parents:
            self.value = cp.add(self.value, parent.value)

    def get_graident(self, parent):
        return self.graident

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
