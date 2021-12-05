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
import cupy as cp

from .graph import Graph, default_graph

class Node(object):
    
    def __init__(self, *parents, **kargs) -> None:
        # basic attribute
        self.kargs = kargs
        self.graph = kargs.get('graph', default_graph)
        self.need_save = kargs.get('need_save', True)
        self.GPU = True
        self.gen_node_name(**kargs)

        # init graph
        self.parents = list(parents)
        self.children = []

        # forward and backward value
        self.value = None
        self.graident = None

        self.in_degree = 0
        self.out_degree = 0
        self.pecedent = 0
        self.successor = 0
        self.mark = False

        for parent in self.parents:
            parent.children.append(self)

        self.graph.add_node(self)

    
    def gen_node_name(self, **kargs):
        prefix = kargs.get('prefix', "")
        if prefix:
            self.name = kargs.get('name', '{}/{}:{}'.format(prefix, self.__class__.__name__, self.graph.node_count()))
        else:
            self.name = kargs.get('name', '{}:{}'.format(self.__class__.__name__, self.graph.node_count()))
        # TODO: support name scope
    
    def get_parents(self):
        return self.parents
    
    def get_children(self):
        return self.children
    
    def reset_pece_succ(self):
        self.pecedent = self.in_degree
        self.successor = self.out_degree
    
    def set_mark(self):
        self.mark = True
        for parent in self.parents:
            if parent.mark == False:
                parent.set_mark()
    
    def calc_degree(self):
        for parent in self.parents:
            if parent.mark:
                self.in_degree += 1

        for child in self.children:
            if child.mark:
                self.out_degree += 1
    
    def forward(self):
        # compute your precedent
        for node in self.parents:
            if node.value is None:
                node.forward()
        self.compute()
    
    def done(self, forward=True):
        if forward:
            for parent in self.parents:
                parent.successor -= 1
                if parent.successor == 0:
                    parent.vacuum(True)
                    # cp._default_memory_pool.free_all_blocks()
        else:
            self.vacuum(False)
            # cp._default_memory_pool.free_all_blocks()
    
    def vacuum(self, forward=True):

        if forward:
            self.value = None
        else:
            self.graident = None
        # cp._default_memory_pool.free_all_blocks()

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
    
    def get_type(self):
        return self.__class__.__name__
    
class Variable(Node):
    
    def __init__(self, dims, init=False, trainable=False, **kargs) -> None:
        Node.__init__(self, **kargs)
        self.dims = dims
        self.trainable = trainable

        # init value
        if init:
            is_bias = kargs.get('bias', False)
            if is_bias:
                self.value = cp.zeros(dims, dtype=cp.int32)
            else:
                mean = kargs.get('mean', 0.0)
                std = kargs.get('std', 0.001)
                self.value = cp.random.normal(loc=mean, scale=std, size=dims, dtype=cp.float32)
    
    def set_value(self, value):
        assert value.shape == self.dims, '{} {} {}'.format(self.name, value.shape, self.dims)

        # or maybe reset value explicitly is better?
        self.reset_value(True)
        self.value = value

    def vacuum(self, forward=True):
        return
        # if self.trainable:
        #     return

        # if forward:
        #     self.value = None
        # else:
        #     self.graident = None