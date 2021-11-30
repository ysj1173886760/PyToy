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

def Dense(input, feature_in, feature_out):
    """[Fully Connected Layer]

    Args:
        input ([ndarray]): [input neurons]
        feature_in ([int]): [feature in]
        feature_out ([int]): [feature out]
    """

    weight = Variable((feature_in, feature_out), init=True, trainable=True)
    bias = Variable((1, feature_out), init=True, trainable=True, bias=True)
    return Add(MatMul(input, weight), bias)