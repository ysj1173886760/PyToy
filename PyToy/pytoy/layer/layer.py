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
    bias = Variable((1, feature_out), init = True, trainable = True, bias = True, prefix = name)
    return AddOperator(MatMulOperator(input, weight, prefix=name), bias, prefix=name)

def Conv(input, channel_in, channel_out, kernel_size, stride, padding, bias=True, **kargs):
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

    if bias:
        # because input is [N, C, H, W], so we boardcast in (0, 2, 3) dims
        bias = Variable((1, channel_out, 1, 1), init=True, trainable=True, bias=True, prefix=name)
        return AddOperator(ConvOperator(input, weight, kernel_size=kernel_size, channel_in=channel_in, channel_out=channel_out,
                                stride=stride, padding=padding, prefix=name), bias, prefix=name)
    else:
        return ConvOperator(input, weight, kernel_size=kernel_size, channel_in=channel_in, channel_out=channel_out,
                                stride=stride, padding=padding, prefix=name)

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

def AvgPooling(input, kernel_size, stride, **kargs):
    """[summary]
    input should be formatted as [N, C, H, W]
    Args:
        input ([type]): [description]
        kernel_size ([type]): [description]
        stride ([type]): [description]
    """

    name = kargs.get('name', "")
    return AvgPoolingOperator(input, kernel_size=kernel_size, stride=stride, prefix=name)

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
    gamma.set_value(np.ones(gamma.dims))

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

def append_namescope(name, scope):
    if name == "":
        return ""
    else:
        return '{}/{}'.format(name, scope)

# referring to https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
def BasicBlock(input, in_channels, out_channels, stride=1, **kargs):
    """[Basic Residual Block]

    Args:
        input ([type]): [description]
        in_channels ([type]): [description]
        out_channels ([type]): [description]
        stride (int, optional): [description]. Defaults to 1.
    """
    name = kargs.get('name', "")
    
    conv1 = Conv(input, in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, 
                 name=append_namescope(name, 'conv1'))
    bn1 = BatchNorm(conv1, name=append_namescope(name, 'bn1'))
    relu1 = ReLU(bn1, name=append_namescope(name, 'relu1'))
    conv2 = Conv(relu1, out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                 name=append_namescope(name, 'conv2'))
    bn2 = BatchNorm(conv2, name=append_namescope(name, 'bn2'))
    residual_function = bn2

    shortcut = input
    if stride != 1 or in_channels != out_channels:
        conv3 = Conv(input, in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False, 
                    name=append_namescope(name, 'conv3'))
        bn3 = BatchNorm(conv3, name=append_namescope(name, 'bn3'))
        shortcut = bn3
    
    return ReLU(AddOperator(residual_function, shortcut, prefix=name), name=append_namescope(name, 'relu3'))

def RNN(inputs, input_size, batch_size=10, hidden_size=10, **kargs):
    """[Recurrent Neural Network]

    Args:
        input ([list]): list element Variable (batch_size, input_size)
        batch_size : not use
        hidden_size
    """
    name = kargs.get('name', "")
    mean = kargs.get('mean', 0.0)
    std = kargs.get('std', 0.001)
    
    U = Variable(dims = (input_size, hidden_size), init=True, trainable=True, std=std, mean=mean, prefix=name)
    W = Variable(dims = (hidden_size, hidden_size), init=True, trainable=True, std=std, mean=mean, prefix=name)
    b = Variable(dims = (1, hidden_size), init=True, trainable=True, bias=True, prefix=name)
    
    last_step = None
    for iv in inputs:
        h = AddOperator(MatMulOperator(iv, U), b)
        if last_step is not None:
            h = AddOperator(MatMulOperator(last_step, W), h)
        h = ReLUOperator(h)
        last_step = h

    return last_step

def LSTM(inputs, input_size, batch_size=10, hidden_size=10, **kargs):
    """[Long Short-Term Memory]

    Args:
        input ([list]): list element Variable (batch_size, input_size)
        hidden_size
    """
    name = kargs.get('name', "")
    mean = kargs.get('mean', 0.0)
    std = kargs.get('std', 0.001)

    # input gate
    Wi = Variable(dims = (input_size, hidden_size), init=True, trainable=True, std=std, mean=mean, prefix=name)
    Ui = Variable(dims = (hidden_size, hidden_size), init=True, trainable=True, std=std, mean=mean, prefix=name)
    bi = Variable(dims = (batch_size, hidden_size), init = True, trainable = True, bias = True, prefix = name)
    # forget gate
    Wf = Variable(dims = (input_size, hidden_size), init=True, trainable=True, std=std, mean=mean, prefix=name)
    Uf = Variable(dims = (hidden_size, hidden_size), init=True, trainable=True, std=std, mean=mean, prefix=name)
    bf = Variable(dims = (batch_size, hidden_size), init = True, trainable = True, bias = True, prefix = name)
    # output gate
    Wo = Variable(dims = (input_size, hidden_size), init=True, trainable=True, std=std, mean=mean, prefix=name)
    Uo = Variable(dims = (hidden_size, hidden_size), init=True, trainable=True, std=std, mean=mean, prefix=name)
    bo = Variable(dims = (batch_size, hidden_size), init = True, trainable = True, bias = True, prefix = name)
    # state
    Wc = Variable(dims = (input_size, hidden_size), init=True, trainable=True, std=std, mean=mean, prefix=name)
    Uc = Variable(dims = (hidden_size, hidden_size), init=True, trainable=True, std=std, mean=mean, prefix=name)
    bc = Variable(dims = (batch_size, hidden_size), init=True, trainable=True, bias=True, prefix=name)

    h = c = None
    for iv in inputs:
        if h is None:
            i = SigmoidOperator(AddOperator(MatMulOperator(iv, Wi), bi))
            f = SigmoidOperator(AddOperator(MatMulOperator(iv, Wf), bf))
            o = SigmoidOperator(AddOperator(MatMulOperator(iv, Wo), bo))
            c = DotOperator(i, TanhOperator(AddOperator(MatMulOperator(iv, Wc), bc)))
            h = DotOperator(o, TanhOperator(c))
        else:
            i = SigmoidOperator(AddOperator(AddOperator(MatMulOperator(iv, Wi), MatMulOperator(h, Ui))), bi)
            f = SigmoidOperator(AddOperator(AddOperator(MatMulOperator(iv, Wf), MatMulOperator(h, Uf))), bf) 
            o = SigmoidOperator(AddOperator(AddOperator(MatMulOperator(iv, Wo), MatMulOperator(h, Uo))), bo)
            c_hat = TanhOperator(AddOperator(MatMulOperator(iv, Wc), MatMulOperator(h, Uc)))
            c = AddOperator(AddOperator(DotOperator(f, c), DotOperator(i, c_hat)), bc)
            h = DotOperator(o, TanhOperator(c))

    return h