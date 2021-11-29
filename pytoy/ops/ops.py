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

    def compute(self):
        self.value = cp.zeros(self.parents[0].shape())
        for parent in self.parents:
            self.value = cp.add(self.value, parent.value)

    def get_graident(self, parent):
        return self.graident

class MatMul(Operator):
    
    def compute(self):
        self.value = cp.matmul(self.parents[0].value, self.parents[1].value)

    def get_graident(self, parent):
        # (m, n) = (m, p) * (p, n)
        if parent is self.parents[0]:
            return cp.matmul(self.graident, self.parents[1].value.T)
        else:
            return cp.matmul(self.parents[0].value.T, self.graident)
