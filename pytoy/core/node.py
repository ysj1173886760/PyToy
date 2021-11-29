# -*- encoding: utf-8 -*-
'''
@File    :   node.py
@Time    :   2021/11/29 19:07:14
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   Basic Node
'''

import abc
import numpy as cp

from .graph import Graph, default_graph

class Node(object):
    
    def __init__(self, *parents, **kargs) -> None:
        # basic attribute
        self.kargs = kargs
        self.graph = kargs.get('graph', default_graph)
        self.need_save = kargs.get('need_save', True)
        self.gen_node_name(**kargs)

        # init graph
        self.parents = list(parents)
        self.children = []

        # forward and backward value
        self.value = None
        self.graident = None

        for parent in self.parents:
            parent.children.append(self)

    
    def gen_node_name(self, **kargs):
        self.name = kargs.get('name', '{}:{}'.format(self.__class__.__name__, self.graph.node_count()))
        # TODO: support name scope
    
    def get_parents(self):
        return self.parents
    
    def get_children(self):
        return self.children
    
    def forward(self):
        # compute your precedent
        for node in self.parents:
            if node.value is None:
                node.forward()
        self.compute()

    @abc.abstractmethod
    def compute(self):
        # overwrite this method
        pass

    @abc.abstractmethod
    def get_graident(self, parent):
        # overwrite this method
        pass
    
    def backward(self, target):
        if self is target:
            return

        if self.graident is None:
            self.graident = cp.zeros_like(self.value)
            for node in self.children:
                if node.value is not None:
                    if node.graident is None:
                        node.backward(target)
                    self.graident += node.get_graident(self)

        return self.graident
    
    def clear_graident(self):
        self.graident = None
    
    def shape(self):
        return self.value.shape
    
    def reset_value(self, recursive=True):
        if self.value is None:
            return

        self.value = None
        if recursive:
            for node in self.children:
                node.reset_value(recursive)
    
class Variable(Node):
    
    def __init__(self, dims, init=False, trainable=False, **kargs) -> None:
        Node.__init__(self, **kargs)
        self.dims = dims
        self.trainable = trainable

        # init value
        if init:
            mean = kargs.get('mean', 0.0)
            std = kargs.get('std', 0.001)
            self.value = cp.random.normal(loc=mean, scale=std, size=dims)
    
    def set_value(self, value):
        assert value.shape == self.dims

        # or maybe reset value explicitly is better?
        self.reset_value(True)
        self.value = value
