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
import cupy as cp

from pytoy.core.core import get_node_from_graph

class Trainer(object):
    
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.target = optimizer.target
        self.graph = self.target.graph
        self.initialize()
    
    def initialize(self):
        # remove useless nodes
        self.graph.reset_mark()
        self.graph.set_mark(self.target)

        # re-calc the degree affected by the marked nodes
        self.graph.calc_degree()

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
            