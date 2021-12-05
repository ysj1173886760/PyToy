# -*- encoding: utf-8 -*-
'''
@File    :   core.py
@Time    :   2021/11/29 21:11:34
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   None
'''

from pytoy.ops.ops import BatchNormOperator
from .node import Variable
from .node import default_graph
import cupy as cp

def get_trainable_variables_from_graph(node_name=None, graph=None):
    if graph is None:
        graph = default_graph
    if node_name is None:
        return [node for node in graph.nodes if isinstance(node, Variable) and node.trainable]

def get_trainable_variables_with_mark(node_name=None, graph=None):
    if graph is None:
        graph = default_graph
    if node_name is None:
        return [node for node in graph.nodes if isinstance(node, Variable) and node.trainable and node.mark]

def get_bn_nodes_with_mark(graph=None):
    if graph is None:
        graph = default_graph
    return [node for node in graph.nodes if isinstance(node, BatchNormOperator) and node.mark]

def get_node_from_graph(node_name, graph=None):
    if graph is None:
        graph = default_graph
    
    return graph.mapping.get(node_name, None)

def update_node_value(value_dict, graph=None):
    if graph is None:
        graph = default_graph

    for name, value in value_dict.items():
        node = get_node_from_graph(name, graph=graph)
        if node.GPU:
            node.value = cp.array(value)
        else:
            node.value = value

def update_node_graident(graident_dict, graph=None):
    if graph is None:
        graph = default_graph

    for name, graident in graident_dict.items():
        node = get_node_from_graph(name, graph=graph)
        if node.GPU:
            node.graident = cp.array(graident)
        else:
            node.graident = graident