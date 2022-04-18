# -*- encoding: utf-8 -*-
'''
@File    :   trainer.py
@Time    :   2021/12/04 13:39:38
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   None
'''

import queue
import numpy as np

from pytoy.core.core import get_bn_nodes_with_mark, get_node_from_graph, get_trainable_variables_with_mark

class Trainer(object):
    
    def __init__(self, optimizer, **kargs) -> None:
        self.optimizer = optimizer
        self.target = optimizer.target
        self.graph = self.target.graph
        self.initialize()
        # only update the nodes with mark
        self.optimizer.trainable_nodes = get_trainable_variables_with_mark(graph=self.graph)
        self.bn_nodes = get_bn_nodes_with_mark(self.graph)
    
    def initialize(self):
        # remove useless nodes
        self.graph.reset_mark()
        self.graph.set_mark(self.target)

        # re-calc the degree affected by the marked nodes
        self.graph.calc_degree()
    
    def get_graidents(self):
        node_graidents = {}
        for node in self.optimizer.trainable_nodes:
            if node.GPU:
                node_graidents[node] = np.array(node.graident.get())
            else:
                node_graidents[node] = np.array(node.graident)
        
        for node in self.bn_nodes:
            if node.GPU:
                var = node.running_var_x.get()
                mean = node.running_mean_x.get()
            else:
                var = node.running_var_x
                mean = node.running_mean_x

            node_graidents[node] = np.concatenate((np.expand_dims(mean, axis=0), np.expand_dims(var, axis=0)), axis=0)

        return node_graidents
    
    def _variable_weights_init(self):
        pass

    def init(self):
        self._variable_weights_init()
    
    def _optimizer_update(self):
        self.optimizer.update()
        
    def update(self):
        self._optimizer_update()

    def train(self, input):
        self.graph.clear_graident()

        self.graph.reset_pece_succ()
        q = queue.Queue()
        # feed the graph
        for node_name in input:
            node = get_node_from_graph(node_name, graph=self.graph)

            if node.mark:
                node.set_value(input[node_name])
                q.put(node)
        
        # normally, the beginging of the graph is input, but it also can be constants
        for node in self.graph.nodes:
            if node.mark and input.get(node.name, None) is None and node.pecedent == 0:
                q.put(node)
        
        # start compute
        while not q.empty():
            cur_node = q.get()
            if cur_node.value is None:
                cur_node.compute()
                cur_node.done(True)

            if cur_node == self.target:
                break
                
            for child in cur_node.children:
                if child.mark == False:
                    continue

                child.pecedent -= 1
                if child.pecedent == 0:
                    q.put(child)
        
        self.graph.reset_pece_succ()

        q = queue.Queue()
        q.put(self.target)

        while not q.empty():
            cur_node = q.get()
            for parent in cur_node.parents:
                if parent.graident is None:
                    parent.graident = cur_node.get_graident(parent)
                else:
                    parent.graident += cur_node.get_graident(parent)

                parent.successor -= 1
                if parent.successor == 0:
                    q.put(parent)

            cur_node.done(False)
            