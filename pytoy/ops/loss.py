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