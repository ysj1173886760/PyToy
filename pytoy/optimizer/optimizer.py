# -*- encoding: utf-8 -*-
'''
@File    :   optimizer.py
@Time    :   2021/12/01 18:07:01
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   None
'''

from pytoy.core.core import get_trainable_variables_from_graph
import abc
import cupy as cp

class Optimizer(object):
    
    def __init__(self, graph, target, lr) -> None:
        """[summary]

        Args:
            graph ([Graph]): [compute graph]
            target ([Node]): [loss that you want to optimize]
            lr ([float]): [learning rate]
        """
        self.graph = graph
        self.target = target
        self.lr = lr
        self.trainable_nodes = get_trainable_variables_from_graph(graph=self.graph)

    def step(self):
        """[do forward and backward]
        """
        self.graph.clear_graident()

        self.target.forward()

        for node in self.trainable_nodes:
            if node.trainable:
                node.backward(self.target)
    
    @abc.abstractmethod
    def _update(self):
        pass
    
    def update(self):
        self._update()
    
    def set_lr(self, lr):
        self.lr = lr

class StochasticGraidentDescent(Optimizer):

    def _update(self):
        for node in self.trainable_nodes:
            if node.trainable:
                node.set_value(node.value - self.lr * node.graident)
    
class Momentum(Optimizer):
    
    def __init__(self, graph, target, lr, momentum=0.9) -> None:
        Optimizer.__init__(self, graph, target, lr)
        self.momentum = momentum
        self.v = {}
    
    def _update(self):
        for node in self.trainable_nodes:
            if node.trainable:
                if node not in self.v:
                    self.v[node] = cp.zeros_like(node.graident)
                self.v[node] = self.momentum * self.v[node] - self.lr * node.graident
                node.set_value(node.value + self.v[node])

class AdaGrad(Optimizer):
    
    def __init__(self, graph, target, lr) -> None:
        Optimizer.__init__(self, graph, target, lr)
        self.s = {}
    
    def _update(self):
        for node in self.trainable_nodes:
            if node.trainable:
                if node not in self.s:
                    self.s[node] = cp.zeros_like(node.graident)
                self.s[node] += cp.power(node.graident, 2)
                node.set_value(node.value - self.lr * node.graident / cp.sqrt(self.s[node] + 1e-9))

class RMSProp(Optimizer):
    
    def __init__(self, graph, target, lr, beta=0.9) -> None:
        Optimizer.__init__(self, graph, target, lr)
        self.s = {}

        assert 0.0 < beta < 1.0
        self.beta = beta

    def _update(self):
        for node in self.trainable_nodes:
            if node.trainable:
                if node not in self.s:
                    self.s[node] = cp.power(node.graident, 2)
                else:
                    self.s[node] = self.beta * self.s[node] + (1 - self.beta) * cp.power(node.graident, 2)
                node.set_value(node.value - self.lr * node.graident / cp.sqrt(self.s[node] + 1e-9))

class Adam(Optimizer):
    
    def __init__(self, graph, target, lr, beta1=0.9, beta2=0.99) -> None:
        Optimizer.__init__(self, graph, target, lr)
        assert 0.0 < beta1 < 1.0
        assert 0.0 < beta2 < 1.0

        self.beta1 = beta1
        self.beta2 = beta2
        self.iter_step = 0

        self.v = {}
        self.s = {}
    
    def _update(self):
        for node in self.trainable_nodes:
            if node.trainable:
                if node not in self.v:
                    # which one is correct? or better?
                    # self.v[node] = node.graident
                    # self.s[node] = cp.power(node.graident, 2)
                    self.v[node] = (1 - self.beta1) * node.graident
                    self.s[node] = (1 - self.beta2) * cp.power(node.graident, 2)
                else:
                    self.v[node] = cp.add(self.beta1 * self.v[node], (1 - self.beta1) * node.graident)
                    self.s[node] = cp.add(self.beta2 * self.s[node], (1 - self.beta2) * cp.power(node.graident, 2))

                self.iter_step += 1
                vt_hat = cp.divide(self.v[node], (1 - cp.power(self.beta1, self.iter_step)))
                st_hat = cp.divide(self.s[node], (1 - cp.power(self.beta2, self.iter_step)))
                node.set_value(node.value - self.lr * vt_hat / cp.sqrt(st_hat + 1e-9))