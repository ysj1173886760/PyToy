# -*- encoding: utf-8 -*-
'''
@File    :   core.py
@Time    :   2021/11/29 21:11:34
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   None
'''

from .node import Variable
from .node import default_graph
import cupy as cp

def get_trainable_variables_from_graph(node_name=None, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if node_name is None:
        return [node for node in default_graph.nodes if isinstance(node, Variable) and node.trainable]

def get_trainable_variables_with_mark(node_name=None, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if node_name is None:
        return [node for node in default_graph.nodes if isinstance(node, Variable) and node.trainable and node.mark]

def get_node_from_graph(node_name, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = name_scope + '/' + node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None

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
            node.value = cp.array(graident)
        else:
            node.graident = graident