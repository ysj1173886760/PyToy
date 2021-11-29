# -*- encoding: utf-8 -*-
'''
@File    :   graph.py
@Time    :   2021/11/29 19:03:55
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   Compute Graph
'''

class Graph:

    def __init__(self) -> None:
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

    # clear all the graident which computed in this pass
    def clear_graident(self):
        for node in self.nodes:
            # TODO: do some real stuff
            pass
    
    
default_graph = Graph()
