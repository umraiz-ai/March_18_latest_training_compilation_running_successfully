# Copyright 2023 The Deeper-I Authors. All Rights Reserved.
#
# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os, sys
sys.path.append('./src')

import argparse
# import json
import numpy as np

import onnx
import onnx.numpy_helper as helper

import networkx as nx

from tachy_format import *
from graph import sort_graph, get_connected


_DEBUG = False


def get_parser():
    """
    Get the parser.
    :return: parser
    """

    parser = argparse.ArgumentParser(description='Configuration file generator from ONNX')
    parser.add_argument('src_file', type=str,
                        help='Source ONNX file',
                        default='20220908_EfficientNet+DM@W23@C32@O2-FPN+DM@W4@O2_0.onnx')

    parser.add_argument('dst_file', type=str,
                        help='Destination directory path',
                        default='.')

    return parser


class ONNXtoTACHY:
    def __init__(self, onnx_model):
        self.model = onnx_model
        self.graph = None
        self.name_table = {}

        self.support_ops = {
            "Conv":True,
            "ConvTranspose":True,
            "BatchNormalization":True,
            "Gemm":True,
            "MatMul":True,
            "Add":True,
            "Reshape":True,

            "LeakyRelu":False,
            "Relu":False,
            "Concat":False,
            # "Add":False,
            "GlobalAveragePool":False,
            "MaxPool":False,
            # "Flatten":False,
            # "Identity":False,
            # "Squeeze":False,
            # "Unsqueeze":False,
            # "Transpose":False,
        }

        self.extra_params = []


    #################################
    ####### Internal
    #################################
    def _get_node_name(self, idx:int) -> str:
        result = None
        for k, v in self.name_table.items():
            if v == idx:
                result = k
                break
        return result

    def _get_node_info(self, name:str):
        result = None 
        for n in self.model.graph.node:
            if n.name == name:
                result = n
    
        return result

    def _search_children(self, t) -> list:
        children = []
        for t_out in t.output:
            for n in self.model.graph.node:
                for t_in in n.input:
                    if t_out == t_in:
                        children.append(n)
    
        return children

    def _get_info_by_idx(self, idx:int):
        return self._get_node_info(self._get_node_name(idx))

    def _remove_prefix(self, name):
        '''
        Sequence is important.
        '''
        name = name.replace('InsertedCast_StatefulPartitionedCall', '')
        name = name.replace('StatefulPartitionedCall', '')
        return name

    def _check_valid_operation(self, node) -> bool:
        if node.op_type in self.support_ops.keys():
            return True
        else:
            return False
                
    def _get_params(self, n_info) -> list:
        result = []
        if n_info.op_type in self.support_ops.keys():
            if self.support_ops[n_info.op_type]:
                result = self._get_params_by_op(n_info)
                # assert len(n_info.input) - 1 == len(result)
                if len(n_info.input) - 1 != len(result):
                    print('[WRAN]: {}, number of parameters are different. {} =! {}'.format(n_info.name, len(n_info.input), len(result)))
                # print(n_info.name, n_info.op_type, len(result))
        return result

    def _get_attribute(self, n_info) -> dict:
        result = {}
        for n_ab in n_info.attribute:
            if n_ab.type == 7: # INTS
                result[n_ab.name] = list(n_ab.ints)
            elif n_ab.type == 2: # INT
                result[n_ab.name] = n_ab.i
            elif n_ab.type == 3: # STRING
                result[n_ab.name] = n_ab.s
            elif n_ab.type == 1: # FLOAT
                result[n_ab.name] = n_ab.f

        return result

    # def _get_configs(self, n_info, params:list) -> dict:
    def _get_configs(self, n_info, params:list, idx:int) -> dict:
        result = {}
        result['inputs'] = list(n_info.input) if len(params) == 0 else list(n_info.input[:len(n_info.input) - len(params)])
        result['outputs'] = list(n_info.output)
        result['configs'] = self._get_attribute(n_info)
        result['op_type'] = "AddP" if n_info.op_type == "Add" and len(params) > 0 else n_info.op_type
        if result['op_type'] == "AddP":
            params = [np.ones_like(params[0]), params[0]]
        result['params'] = params
            
        result['name'] = '{}_{}'.format(n_info.op_type, idx)

        return result

    def _get_params_identity(self, n_info) -> list:
        '''
        Support Identity node with single input and output.
        '''
        result = {}
        if n_info.op_type == 'Identity':
            params = self._get_params_by_op(n_info, extra=False)
            for iden_out, param in zip(n_info.output, params):
                result.update({iden_out:param})
        return result

    def _get_params_by_op(self, n_info, extra=True) -> list:
        result = []
        for _input in n_info.input:
            for n in self.model.graph.initializer:
                if self._remove_prefix(n.name) == self._remove_prefix(_input):
                    result.append(helper.to_array(n))
                    break

            if extra:
                for name, param in self.extra_params.items():
                    if self._remove_prefix(name) == self._remove_prefix(_input):
                        result.append(param)
                        break

        # print(n_info.op_type, len(result))
        return result

    def _config_locked_dimension(self, graph:nx.DiGraph, locked_dim=None) -> nx.DiGraph:
        for i, n_idx in enumerate(graph.nodes):
            graph.nodes[n_idx]['configs']['locked_dim'] = locked_dim 
        return graph

    def _config_op_dimension(self, graph:nx.DiGraph, init_op_dim=2) -> nx.DiGraph:
        op_dim_pre = init_op_dim
        for i, n_idx in enumerate(graph.nodes):
            if graph.nodes[n_idx]['op_type'] == 'Conv':
                if len(graph.nodes[n_idx]['params'][0].shape) == 3:
                    graph.nodes[n_idx]['configs']['op_dim'] = 1 
                elif len(graph.nodes[n_idx]['params'][0].shape) == 4:
                    graph.nodes[n_idx]['configs']['op_dim'] = 2 
                op_dim_pre = graph.nodes[n_idx]['configs']['op_dim']

            elif graph.nodes[n_idx]['op_type'] == 'MaxPool':
                if len(graph.nodes[n_idx]['configs']['kernel_shape']) == 1:
                    graph.nodes[n_idx]['configs']['op_dim'] = 1 
                elif len(graph.nodes[n_idx]['configs']['kernel_shape']) == 2:
                    graph.nodes[n_idx]['configs']['op_dim'] = 2 
                op_dim_pre = graph.nodes[n_idx]['configs']['op_dim']

            elif graph.nodes[n_idx]['op_type'] == 'ConvTranspose':
                if len(graph.nodes[n_idx]['params'][0].shape) == 3:
                    graph.nodes[n_idx]['configs']['op_dim'] = 1 
                elif len(graph.nodes[n_idx]['params'][0].shape) == 4:
                    graph.nodes[n_idx]['configs']['op_dim'] = 2 
                op_dim_pre = graph.nodes[n_idx]['configs']['op_dim']

            elif graph.nodes[n_idx]['op_type'] == 'AddP':
                if len(graph.nodes[n_idx]['params'][0].shape) == 3:
                    graph.nodes[n_idx]['configs']['op_dim'] = 1 
                elif len(graph.nodes[n_idx]['params'][0].shape) == 4:
                    graph.nodes[n_idx]['configs']['op_dim'] = 2 
                op_dim_pre = graph.nodes[n_idx]['configs']['op_dim']

            elif graph.nodes[n_idx]['op_type'] == 'GlobalAveragePool' or \
                    graph.nodes[n_idx]['op_type'] == 'BatchNormalization':
                graph.nodes[n_idx]['configs']['op_dim'] = op_dim_pre

        return graph

    def _find_float_input_index(self, graph, n_idx:int) -> int:
        edges = graph.edges
        candidates = []
        for ie, oe in edges:
            if oe == n_idx:
                candidates += graph.nodes[ie]["outputs"]

        float_list = list(set(graph.nodes[n_idx]["inputs"]) - set(candidates))
        assert len(float_list) == 1
        return graph.nodes[n_idx]["inputs"].index(float_list[0])

    def _find_float_output_index(self, graph, n_idx:int) -> int:
        edges = graph.edges
        candidates = []
        for ie, oe in edges:
            if ie == n_idx:
                candidates += graph.nodes[oe]["inputs"]

        float_list = list(set(graph.nodes[n_idx]["outputs"]) - set(candidates))
        assert len(float_list) == 1
        return graph.nodes[n_idx]["outputs"].index(float_list[0])


    #################################
    ####### Internal - Main
    #################################
    def _convert_graph(self):
        dg = nx.DiGraph()
        for i, p_node in enumerate(self.model.graph.node):
            if p_node.name not in self.name_table.keys():
                self.name_table[p_node.name] = len(self.name_table) + 1
            children = self._search_children(p_node)
            for c_node in children:
                if c_node.name not in self.name_table.keys():
                    self.name_table[c_node.name] = len(self.name_table) + 1
                dg.add_edge(self.name_table[p_node.name], self.name_table[c_node.name])

        # Single depth
        if len(dg.nodes) == 0:
            for _, v in self.name_table.items():
                dg.add_node(v)
        assert len(dg.nodes) > 0 and len(self.name_table) > 0

        return dg

    def _update_graph(self, graph:nx.DiGraph):
        for n_idx in graph.nodes:
            n_info = self._get_info_by_idx(n_idx)
            params = self._get_params(n_info)
            # cfgs = self._get_configs(n_info, params)
            cfgs = self._get_configs(n_info, params, n_idx)
            graph.nodes[n_idx].update(cfgs)
        return graph

    def _identity_params(self, graph:nx.DiGraph):
        result = {}
        for i in graph.nodes:
            n_info = self._get_info_by_idx(i)
            n_dict = self._get_params_identity(n_info)
            result.update(n_dict)
        return result

    # TODO: need fn for multiple input and output (support 1:1 now)
    def _refresh_inputs(self, graph:nx.DiGraph) -> nx.DiGraph:
        edges = graph.edges
        for ie, oe in edges:
            outputs = graph.nodes[ie]["outputs"]
            inputs = graph.nodes[oe]["inputs"]
            if len(set(outputs) & set(inputs)) == 0:
                iie = self._find_float_output_index(graph, ie)
                ooe = self._find_float_input_index(graph, oe)
                graph.nodes[oe]["inputs"][ooe] = outputs[iie]
                # graph.nodes[oe]["inputs"] = outputs

        return graph

    #################################
    ####### External - Methods
    #################################
    def shrink_graph(self, init_op_dim=2, locked_dim=None):
        # Convert ModelProto -> DiGraph
        dg = self._convert_graph()

        # Prepare to convert
        self.extra_params = self._identity_params(dg)

        # Create New DiGraph
        dg_new = nx.DiGraph()
        for n_idx in dg.nodes:
            n_name = self._get_node_name(n_idx)
            node = self._get_node_info(n_name)
            # TODO: empty input, empty output
            vld_op = self._check_valid_operation(node)
            vld = vld_op
            if vld: dg_new.add_node(n_idx)
            
        if len(dg.edges) > 0:
            edges = []
            for n_idx in dg_new.nodes:
                for o_idx in get_connected(n_idx, dg, dg_new.nodes):
                    edges.append((n_idx, o_idx,))
            dg_new.add_edges_from(edges)
            graph = self._update_graph(sort_graph(dg_new, edges))
            graph = self._refresh_inputs(graph)

        else:
            graph = self._update_graph(dg_new)

        self.graph = self._config_op_dimension(graph, init_op_dim)
        self.graph = self._config_locked_dimension(graph, locked_dim)

        return self


    #################################
    ####### External - Display
    #################################
    # ONNX fn
    def show_inputs(self):
        for i, n in enumerate(self.model.graph.input):
            print(n)
    
    def show_outputs(self):
        for i, n in enumerate(self.model.graph.output):
            print(n)
    
    def show_nodes(self):
        for i, n in enumerate(self.model.graph.node):
            print(n.name, n)
    
    def show_params(self):
        for i, n in enumerate(self.model.graph.initializer):
            print(n.name, helper.to_array(n).shape)



if __name__ == '__main__':
    args = get_parser().parse_args()
    src_file = args.src_file
    dst_file = args.dst_file

    tm = ONNXtoTACHY(onnx.load(src_file))
    tm = tm.shrink_graph()

    tachy_dict = tdict()
    tachy_dict['tachy_layer_model'] = tm
    tsave(dst_file, tachy_dict)
    # cfg = tm.get_param_config()
    # print(json.dumps(cfg, indent=4))

