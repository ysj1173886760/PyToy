# -*- encoding: utf-8 -*-
'''
@File    :   ps.py
@Time    :   2021/12/05 13:02:14
@Author  :   sheep 
@Version :   1.0
@Contact :   1173886760@qq.com
@Desc    :   None
'''

import grpc
import numpy as np
import cupy as cp
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ..proto import parameter_server_pb2
from ..proto import parameter_server_pb2_grpc
from ..distribute import DistributeCommon

class ParameterService(parameter_server_pb2_grpc.ParameterServiceServicer):
    
    def __init__(self, worker_num, sync=True):
        self.node_graidents_cache = {}
        self.variable_weights_cache = {}

        self.sync = sync
        self.worker_num = worker_num

        self.cur_push_num = 0
        self.cur_pull_num = self.worker_num
        self.batch_num = 0

        self.cond = threading.Condition()
        self.init_lock = threading.Lock()

        # first one won
        self.is_init = False
    
    def Push(self, push_req, context):
        node_graidents_dict = DistributeCommon._deserialize_proto_node_graidents(push_req.node_graidents)
        
        if self.sync:
            self._push_sync(node_graidents_dict)
        else:
            self._push_async(node_graidents_dict)

        return parameter_server_pb2.ParameterPullResp()
    
    def _push_sync(self, node_graidents_dict):
        if self.cond.acquire():
            # wait until all workers has pull the parameter
            while self.cur_pull_num != self.worker_num:
                self.cond.wait()

            self.cur_push_num += 1
            self.batch_num += 1
            self._update_graidents_cache(node_graidents_dict)

            if self.cur_push_num >= self.worker_num:
                self.cur_pull_num = 0
                # notify all workers who is blocked when pulling parameter
                self.cond.notify_all()

    def _push_async(self, node_graidents_dict):
        raise NotImplementedError()
    
    def Pull(self, pull_req, context):
        # ignore the pull_req, and return all of the graidents
        if self.sync:
            return self._pull_sync()
        else:
            return self._pull_async()
        

    def _pull_sync(self):
        if self.cond.acquire():
            # this clause is the same as wait_for(self.cur_push_num != self.worker_num) ?
            # in cpp, this assert is true. Not sure it's the same in python
            while self.cur_push_num != self.worker_num:
                self.cond.wait()
            
            self.cur_pull_num += 1
            self._calc_mean_graidents()
            resp = self._serialize_pull_resp()

            if self.cur_pull_num >= self.worker_num:
                self.cur_push_num = 0
                self._reset_graidents_cache()
                self.cond.notify_all()

        return resp

    def _pull_async(self):
        raise NotImplementedError()

    def _calc_mean_graidents(self):
        if self.batch_num != 0:
            for name in self.node_graidents_cache:
                self.node_graidents_cache[name] /= self.batch_num

            self.batch_num = 0

    def _update_graidents_cache(self, node_graidents_dict):
        for name, graident in node_graidents_dict.items():
            if name in self.node_graidents_cache:
                self.node_graidents_cache[name] += graident
            else:
                self.node_graidents_cache[name] = graident

    def _serialize_pull_resp(self):
        proto_node_graidents = DistributeCommon._serialize_proto_node_graidents(self.node_graidents_cache)
        resp = parameter_server_pb2.ParameterPullResp(node_graidents=proto_node_graidents)
        return resp

    def _reset_graidents_cache(self):
        self.node_graidents_cache.clear()

    def VariableWeightsInit(self, request, context):
        self.init_lock.acquire()

        if not self.is_init:
            self.variable_weights_cache = DistributeCommon._deserialize_proto_variable_weights(request)
            self.is_init = True
            print('[INIT] Initializing Variable Weights...')
        
        resp = DistributeCommon._serialize_proto_variable_weights(self.variable_weights_cache)

        self.init_lock.release()
        return resp

class ParameterServiceClient(object):
    
    def __init__(self, host) -> None:
        self.stub = parameter_server_pb2_grpc.ParameterServiceStub(grpc.insecure_channel(host))

        assert self.stub is not None
        
        print('[GRPC] Connected to Parameter Server {}'.format(host))
    
    def variable_weights_init(self, var_weights_dict):
        init_req = DistributeCommon._serialize_proto_variable_weights(var_weights_dict)

        init_resp = self.stub.VariableWeightsInit(init_req)

        ps_var_weights = DistributeCommon._deserialize_proto_variable_weights(init_resp)

        return ps_var_weights
    
    def push_graidents(self, graidents):
        proto_node_graidents = DistributeCommon._serialize_proto_node_graidents(graidents)
        push_req = parameter_server_pb2.ParameterPushReq(node_graidents=proto_node_graidents)

        resp = self.stub.Push(push_req)

        return resp
        
    def pull_graidents(self, nodes_name=None):
        # Pull all graidents
        pull_req = parameter_server_pb2.ParameterPullReq()

        pull_resp = self.stub.Pull(pull_req)

        node_graidents = DistributeCommon._deserialize_proto_node_graidents(pull_resp.node_graidents)

        return node_graidents

class ParameterServiceServer(object):
    
    def __init__(self, cluster_conf, sync=True, max_workers=10):
        self.worker_num = len(cluster_conf['workers'])
        self.host = cluster_conf['ps'][0]
        self.sync = sync
        self.max_workers = max_workers

        self.server = grpc.server(ThreadPoolExecutor(max_workers=self.max_workers))
        parameter_server_pb2_grpc.add_ParameterServiceServicer_to_server(
            ParameterService(self.worker_num, self.sync), self.server)
        self.server.add_insecure_port(self.host)
    
    def serve(self):
        self.server.start()
        print('[PS] Parameter server (mode: {}) running on {} worker num {}'.format('Sync' if self.sync else 'Async',
                                                                                    self.host, self.worker_num))
        
        try: 
            while True:
                time.sleep(60 * 60 * 24)
        except KeyboardInterrupt:
            # same as None?
            self.server.stop(0)
