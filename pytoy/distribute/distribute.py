# -*- encoding: utf-8 -*-
'''
@File    :   distribute.py
@Time    :   2021/12/05 10:06:19
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   None
'''

import numpy as np
import cupy as cp

from ..core import Node
from .proto import common_pb2
from pytoy.distribute import proto

class DistributeCommon(object):
    @staticmethod
    def _serialize_proto_node_graidents(node_graidents_dict, to_server=True):
        proto_node_graidents = common_pb2.NodeGraidents()
        if to_server:
            for node, g in node_graidents_dict.items():
                proto_node = proto_node_graidents.nodes.add()
                proto_node.name = node.name
                proto_node.node_type = node.get_type()

                proto_graident = proto_node_graidents.graidents.add()
                proto_graident.value.extend(np.array(g).flatten())
                proto_graident.dims.extend(list(g.shape))
            
        else:
            for name, g in node_graidents_dict.items():
                proto_node = proto_node_graidents.nodes.add()
                proto_node.name = name

                proto_graident = proto_node_graidents.graidents.add()
                proto_graident.value.extend(np.array(g).flatten())
                proto_graident.dims.extend(list(g.shape))
            
        return proto_node_graidents
            
    
    @staticmethod
    def _deserialize_proto_node_graidents(node_graidents, from_server=True):
        proto_nodes = node_graidents.nodes
        proto_graidents = node_graidents.graidents

        assert len(proto_nodes) == len(proto_graidents)

        node_graidents_dict = {}

        if from_server:
            for index in range(len(proto_nodes)):
                node_name = proto_nodes[index].name
                graident_value = proto_graidents[index].value
                graident_dims = tuple(proto_graidents[index].dims)
                # there could be precision loss?
                node_graidents_dict[node_name] = np.array(graident_value, dtype=np.float32).reshape(graident_dims)
        else:
            for index in range(len(proto_nodes)):
                node_name = proto_nodes[index].name
                node_type = proto_nodes[index].node_type
                graident_value = proto_graidents[index].value
                graident_dims = tuple(proto_graidents[index].dims)
                node_graidents_dict[node_name] = (np.array(graident_value, dtype=np.float32).reshape(graident_dims), node_type)
        
        return node_graidents_dict

    @staticmethod
    def _serialize_proto_variable_weights(variable_weights_dict):
        proto_variable_weights = common_pb2.VariableWeightsReqResp()

        for name, weights in variable_weights_dict.items():
            proto_node = proto_variable_weights.variables.add()
            proto_node.name = name

            proto_weights = proto_variable_weights.weights.add()
            proto_weights.value.extend(np.array(weights).flatten())
            proto_weights.dims.extend(list(weights.shape))
        
        return proto_variable_weights
    
    @staticmethod
    def _deserialize_proto_variable_weights(variable_weights):
        proto_nodes = variable_weights.variables
        proto_weights = variable_weights.weights

        assert len(proto_nodes) == len(proto_weights)

        variable_weights_dict = {}

        for index in range(len(proto_nodes)):
            node_name = proto_nodes[index].name
            weights = proto_weights[index].value
            dims = proto_weights[index].dims
            variable_weights_dict[node_name] = np.array(weights, dtype=np.float32).reshape(dims)

        return variable_weights_dict