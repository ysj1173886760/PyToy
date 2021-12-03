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
import cupy as cp
import numpy as np

def Dense(input, feature_in, feature_out, **kargs):
    """[Fully Connected Layer]

    Args:
        input ([ndarray]): [input neurons]
        feature_in ([int]): [feature in]
        feature_out ([int]): [feature out]
    """
    name = kargs.get('name', "")
    mean = kargs.get('mean', 0.0)
    std = kargs.get('std', 0.001)
    weight = Variable((feature_in, feature_out), init=True, trainable=True, std=std, mean=mean, prefix=name)
    bias = Variable((1, feature_out), init=True, trainable=True, bias=True, prefix=name)
    return AddOperator(MatMulOperator(input, weight, prefix=name), bias, prefix=name)

def Conv(input, channel_in, channel_out, kernel_size, stride, padding, **kargs):
    """[summary]
    input should be formatted as [N, C, H, W]
    
    Args:
        input ([type]): [description]
        feature_in ([type]): [description]
        feature_out ([type]): [description]
        kernel_size ([type]): [description]
        stride ([type]): [description]
        padding ([type]): [description]
    """

    name = kargs.get('name', "")
    mean = kargs.get('mean', 0.0)
    std = kargs.get('std', 0.001)
    # [Cin, k, k, Cout]
    weight = Variable((channel_in, kernel_size, kernel_size, channel_out), init=True, trainable=True, std=std, mean=mean, prefix=name)
    # because input is [N, C, H, W], so we boardcast in (0, 2, 3) dims
    bias = Variable((1, channel_out, 1, 1), init=True, trainable=True, bias=True, prefix=name)
    return AddOperator(ConvOperator(input, weight, kernel_size=kernel_size, channel_in=channel_in, channel_out=channel_out,
                            stride=stride, padding=padding, prefix=name), bias, prefix=name)

def MaxPooling(input, kernel_size, stride, **kargs):
    """[summary]
    input should be formatted as [N, C, H, W]
    Args:
        input ([type]): [description]
        kernel_size ([type]): [description]
        stride ([type]): [description]
    """

    name = kargs.get('name', "")
    return MaxPoolingOperator(input, kernel_size=kernel_size, stride=stride, prefix=name)

def Flatten(input, **kargs):
    """[summary]
    use reshape operator to flatten the input

    Args:
        input ([type]): [description]
    """
    name = kargs.get('name', "")
    batch_size = input.dims[0]
    feature_size = np.product(input.dims[1: ])
    return ReshapeOperator(input, to_shape=(batch_size, feature_size), prefix=name)

def BatchNorm(input, **kargs):
    """[batch normalization layer]

    Args:
        input ([type]): [description]
    """

    name = kargs.get('name', "")
    gamma = Variable(tuple([1] + list(input.dims[1: ])), init=False, trainable=True, prefix=name)
    # initialize to all one matrix
    gamma.set_value(cp.ones(gamma.dims))

    bias = Variable(tuple([1] + list(input.dims[1: ])), init=True, trainable=True, bias=True, prefix=name)
    return BatchNormOperator(input, gamma, bias, prefix=name)

def ReLU(input, **kargs):
    """[ReLU layer]

    Args:
        input ([type]): [description]
    """

    name = kargs.get('name' "")
    return ReLUOperator(input, prefix=name)

def DropOut(input, drop_prob, **kargs):
    """[DropOut layer]

    Args:
        input ([type]): [description]
        drop_prob ([type]): [description]
    """

    name = kargs.get('name', "")
    return DropOutOperator(input, drop_prob=drop_prob, prefix=name)