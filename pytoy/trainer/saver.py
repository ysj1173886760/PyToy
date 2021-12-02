# -*- encoding: utf-8 -*-
'''
@File    :   saver.py
@Time    :   2021/12/02 19:13:59
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   None
'''

import os
import json
import numpy as np
import datetime
import cupy
from pytoy.core.core import get_node_from_graph

from pytoy.core.node import Variable
from ..utils import ClassMining
from ..core import default_graph
from ..core import Node

# referring https://blog.csdn.net/u010452967/article/details/106259505
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(np.obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class Saver(object):
    """[compute graph saver]

    Args:
        object ([type]): [description]
    """

    def __init__(self, root_dir='./') -> None:
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            os.mkdir(self.root_dir)
    
    def save(self, graph=None, 
             model_file_name='model.json',
             weights_file_name='weights.npz'):
        if graph is None:
            graph = default_graph
        
        meta = {}
        meta['save_time'] = str(datetime.datetime.now())
        
        self._save_model_and_weights(graph, meta, model_file_name, weights_file_name)
        print("Saving Model Success")
    
    def _save_model_and_weights(self, graph, meta, model_file_name, weights_file_name):
        model_json = {
            'meta': meta
        }
        graph_json = []
        weights_dict = dict()
        for node in graph.nodes:
            if not node.need_save:
                continue
            node.kargs.pop('name', None)
            node_json = {
                'node_type': node.__class__.__name__,
                'name': node.name,
                'parents': [parent.name for parent in node.parents],
                'children': [child.name for child in node.children],
                'kargs': node.kargs,
            }

            if isinstance(node, Variable):
                node_json['dims'] = node.dims
                node_json['trainable'] = node.trainable

            graph_json.append(node_json)
            if isinstance(node, Variable):
                weights_dict[node.name] = node.value

        model_json['graph'] = graph_json

        model_file_path = os.path.join(self.root_dir, model_file_name)
        with open(model_file_path, 'w') as model_file:
            json.dump(model_json, model_file, indent=4, cls=NpEncoder)
            
        weight_file_path = os.path.join(self.root_dir, weights_file_name)
        with open(weight_file_path, 'wb') as weight_file:
            np.savez(weight_file, **weights_dict)
    
    @staticmethod
    def create_node(graph, from_model_json, node_json):
        node_type = node_json['node_type']
        node_name = node_json['name']
        parents_name = node_json['parents']
        dims = node_json.get('dims', None)
        trainable = node_json.get('trainable', None)
        kargs = node_json.get('kargs', None)

        parents = []
        for parent_name in parents_name:
            parent_node = get_node_from_graph(parent_name, graph=graph)
            if parent_node is None:
                for node in from_model_json:
                    if node['name'] == parent_name:
                        parent_node_json = node
                
                assert parent_node_json is not None
                parent_node = Saver.create_node(graph, from_model_json, parent_node_json)
            
            parents.append(parent_node)
        
        if node_type == 'Variable':
            assert dims is not None
            dims = tuple(dims)
            return ClassMining.get_instance_by_subclass_name(Node, node_type)(*parents, trainable=trainable, dims=dims, name=node_name, **kargs)
        else:
            return ClassMining.get_instance_by_subclass_name(Node, node_type)(*parents, name=node_name, **kargs)
    
    def _restore_node(self, graph, from_model_json, from_weights_dict):

        for index in range(len(from_model_json)):
            node_json = from_model_json[index]
            node_name = node_json['name']

            weights = None
            if node_name in from_weights_dict:
                weights = from_weights_dict[node_name]
            
            target_node = get_node_from_graph(node_name, graph=graph)
            if target_node is None:
                target_node = Saver.create_node(graph, from_model_json, node_json)

            if weights is not None:
                target_node.value = cupy.array(weights)
    
    def load(self, to_graph=None,
             model_file_name='model.json',
             weights_file_name='weights.npz'):
        if to_graph is None:
            to_graph = default_graph
        
        model_json = {}
        graph_json = []
        weights_dict = dict()

        model_file_path = os.path.join(self.root_dir, model_file_name)
        with open(model_file_path, 'r') as model_file:
            model_json = json.load(model_file)
        
        weights_file_path = os.path.join(self.root_dir, weights_file_name)
        with open(weights_file_path, 'rb') as weights_file:
            weights_npz_files = np.load(weights_file)
            for node_name in weights_npz_files:
                weights_dict[node_name] = weights_npz_files[node_name]
            weights_npz_files.close()
        
        graph_json = model_json['graph']
        self._restore_node(to_graph, graph_json, weights_dict)
        print('Loading Model Success')