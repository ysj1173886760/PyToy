# -*- encoding: utf-8 -*-
'''
@File    :   distributed_trainer.py
@Time    :   2021/12/05 14:22:34
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   None
'''

from numpy.lib.arraysetops import isin
from pytoy.core.core import get_trainable_variables_with_mark, update_node_graident, update_node_value
from ..distribute import ps
from .trainer import Trainer

import numpy as np
import cupy as cp

class DistributedTrainerParameterServer(Trainer):
    def __init__(self, optimizer, **kargs) -> None:
        Trainer.__init__(self, optimizer, **kargs)
        cluster_conf = kargs['cluster_conf']
        ps_host = cluster_conf['ps'][0]
        self.ps_client = ps.ParameterServiceClient(ps_host)
    
    def _variable_weights_init(self):
        trainable_nodes = get_trainable_variables_with_mark(graph=self.graph)
        variable_weight_dict = {}
        for node in trainable_nodes:
            if node.GPU:
                variable_weight_dict[node.name] = np.array(node.value.get())
            else:
                variable_weight_dict[node.name] = node.value
        
        ps_variable_weight = self.ps_client.variable_weights_init(variable_weight_dict)
        update_node_value(ps_variable_weight, self.graph)

        print('[INIT] Initialized Worker Variable Weights')
    
    def _optimizer_update(self):
        self.ps_client.push_graidents(self.get_graidents())

        node_graidents_dict = self.ps_client.pull_graidents()

        update_node_graident(node_graidents_dict, self.graph)
        self.optimizer.update()