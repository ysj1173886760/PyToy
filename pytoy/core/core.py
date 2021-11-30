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

def get_trainable_variables_from_graph(node_name=None, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if node_name is None:
        return [node for node in default_graph.nodes if isinstance(node, Variable) and node.trainable]
