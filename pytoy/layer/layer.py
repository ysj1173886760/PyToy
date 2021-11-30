# -*- encoding: utf-8 -*-
'''
@File    :   layer.py
@Time    :   2021/11/30 14:43:25
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   None
'''

from ..core import *
from ..ops import *

def Dense(input, feature_in, feature_out, **kargs):
    """[Fully Connected Layer]

    Args:
        input ([ndarray]): [input neurons]
        feature_in ([int]): [feature in]
        feature_out ([int]): [feature out]
    """
    name = kargs.get('name', "")
    std = kargs.get('std', 0.001)
    weight = Variable((feature_in, feature_out), init=True, trainable=True, std=std, prefix=name)
    bias = Variable((1, feature_out), init=True, trainable=True, bias=True, prefix=name)
    return Add(MatMul(input, weight, prefix=name), bias, prefix=name)