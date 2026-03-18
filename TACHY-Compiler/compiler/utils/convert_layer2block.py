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

import networkx as nx
from ordered_set import OrderedSet as oset

from tachy_format import *
from graph import update_edges, index_by_name, connection_by_edge
from editor import create_node, count_node, identical_kernel, identical_weight, identical_bias
from editor import concat_to_conv, gap_to_conv, split_conv, gap_to_serial_conv, fc_to_conv, align_data, group_conv, reorder_logit, split_maxpool, merge_block, reshape_outputs


_DEBUG = False


def get_parser():
    """
    Get the parser.
    :return: parser
    """

    parser = argparse.ArgumentParser(description='Configuration file generator from TACHY')
    parser.add_argument('src_file', type=str,
                        help='Source TACHY file',
                        default=None)

    parser.add_argument('dst_file', type=str,
                        help='Destination directory path',
                        default=None)
    return parser



class LAYERtoBLOCK:
    def __init__(self, layers):
        self.graph = nx.DiGraph()
        self.refer = layers.graph
        # self.edges = []

        self.npu_ops = {
            0: ["Conv", "ConvTranspose", "Gemm", "MaxPool", "GlobalAveragePool", "Concat", "MatMul"],
            1: ["BatchNormalization", "PRelu", "AddP"],
            2: ["Add"],
            3: ["LeakyRelu", "Relu"],
            4: ["Reshape"], # Soft pipeline
        }

        self.support_ops = [
            "Conv",
            "Concat",
            "IdenticalConv",
            "MaxPool",
            "ConvSplit",
            "Gemm",
            "MatMul",
            "GlobalAveragePool",
            "ConvTranspose",
            "AddP",
        ]

        self.lock_ops = [
            "GlobalAveragePool",
            "MatMul",
            "Gemm",
        ]

        self.cnt = 0
        self.node = None
        self._init_node()

    #################################
    ####### Internal
    #################################
    def _get_block_type(self, op_type:str) -> int:
        blk_type = None
        for k, v in self.npu_ops.items():
            if op_type in v: 
                blk_type = k

        return blk_type

    # def _tuple2list(self, x):
    #     result = [] 
    #     for _x in x:
    #         result.append(list(_x))
    #     return result

    def _init_node(self):
        self.node = create_node(self.cnt)
        return self

    def _count_node(self, target:tuple=None):
        return count_node(self.node, target=target)
            
    def _update_node(self, layer):
        blk_type = self._get_block_type(layer["op_type"])
        self.node["block_function"] = layer["op_type"] if self.node["block_function"] is None else self.node["block_function"]
        self.node["layers"][blk_type].append(layer)
        self.node["configs"].update(layer["configs"])
        self.node["pads_dynamic"] = layer["pads_dynamic"]

        return self

    def _complete(self, layer) -> bool:
        blk_type = self._get_block_type(layer["op_type"])
        if blk_type == 0 and self._count_node() > 0:
            return True
        elif blk_type == 1 and self._count_node(target=(2, 3)) > 0:
            return True
        elif blk_type == 2 and self._count_node(target=(2, 3)) > 0:
            return True
        elif blk_type == 3 and self._count_node(target=(3,)) > 0:
            return True
        elif blk_type == 4 and self._count_node(target=(4,)) > 0: # allow only one reshape component
            return True
        else:
            return False

    def _check_connection(self, layer, blk_list:list) -> bool:
        result = False
        cnt = 0
        for blk_idx in blk_list:
            nodes = self.node["layers"][blk_idx]
            for node in nodes:
                for i_node in layer["inputs"]:
                    if i_node in node["outputs"]:
                        result = True
                        cnt += 1
        return result, cnt

    def _exist_layer(self, n):
        return True if len(self.node["layers"][n]) > 0 else False

    def _disconnection(self, layer) -> bool:
        blk_type = self._get_block_type(layer["op_type"])
        if blk_type == 1:
            if self._exist_layer(1):
                conn, _ = self._check_connection(layer, (1,))
                if not conn: return True
            else:
                conn, _ = self._check_connection(layer, (0,))
                if not conn: return True

        elif blk_type == 2:
            _, cnt = self._check_connection(layer, (0, 1))
            if cnt != 1: return True

        elif blk_type == 3:
            if self._exist_layer(2):
                conn, _ = self._check_connection(layer, (2,))
                if not conn: return True
            elif self._exist_layer(1):
                conn, _ = self._check_connection(layer, (1,))
                if not conn: return True
            elif self._exist_layer(0):
                conn, _ = self._check_connection(layer, (0,))
                if not conn: return True

        elif blk_type == 4:
            if self._exist_layer(3):
                conn, _ = self._check_connection(layer, (3,))
                if not conn: return True
            elif self._exist_layer(2):
                conn, _ = self._check_connection(layer, (2,))
                if not conn: return True
            elif self._exist_layer(1):
                conn, _ = self._check_connection(layer, (1,))
                if not conn: return True
            elif self._exist_layer(0):
                conn, _ = self._check_connection(layer, (0,))
                if not conn: return True
        else:
            return False

    def _update_blocks(self):
        self.graph.add_node(self.cnt)
        self.graph.nodes[self.cnt].update(self.node)
        self.cnt += 1
        self._init_node()
        return self

    def _init_nodes(self):
        for ni in self.refer.nodes:
            layer = self.refer.nodes[ni]
            if self._complete(layer) or self._disconnection(layer): # Last
                self._update_blocks()
            self._update_node(layer)

        self._update_blocks()
        return self

    # def _init_edges(self):
    #     for bi in self.graph.nodes:
    #         inputs, outputs = [], []
    #         input_shapes, output_shapes = [], []
    #         block_inout = oset([])
    #         for _, layers in self.graph.nodes[bi]["layers"].items():
    #             # Unique inout, output
    #             for layer in layers:
    #                 inout = oset(layer["inputs"] + layer["outputs"])
    #                 intersection = block_inout & inout  
    #                 block_inout = (block_inout | inout) - intersection

    #         # Assign input, output
    #         for inout in block_inout:
    #             for _, layers in self.graph.nodes[bi]["layers"].items():
    #                 for layer in layers:
    #                     for _inputs, _shapes in zip(layer["inputs"], layer["input_shapes"]):
    #                         if inout == _inputs:
    #                             inputs.append(_inputs)
    #                             input_shapes.append(_shapes)
    #                             break

    #                     for _outputs, _shapes in zip(layer["outputs"], layer["output_shapes"]):
    #                         if inout == _outputs:
    #                             outputs.append(_outputs)
    #                             output_shapes.append(_shapes)
    #                             break

    #         self.graph.nodes[bi]["inputs"] = inputs
    #         self.graph.nodes[bi]["outputs"] = outputs
    #         self.graph.nodes[bi]["input_shapes"] = input_shapes
    #         self.graph.nodes[bi]["output_shapes"] = output_shapes

    #     # Edge
    #     self.graph = update_edges(self.graph)
    #     return self

    #############
    ##### Global Connection
    #############
    # def _get_part_graph(self, node_names:list) -> nx.DiGraph:
    #     graph = nx.DiGraph()
    #     indices = [index_by_name(self.refer, name) for name in node_names] 
    #     for edge in self.refer.edges:
    #         sn, en = edge
    #         if sn in indices or en in indices:
    #             graph.add_edge(sn, en)
    #             graph.nodes[sn].update(self.refer.nodes[sn])
    #             graph.nodes[en].update(self.refer.nodes[en])
    #     return graph, indices

    # def _get_external_nodes(self, block:list) -> list:
    #     nodes = []
    #     names = []
    #     for _, v in block.items():
    #         nodes += v
    #     names = [node["name"] for node in nodes]
    #     graph, indices = self._get_part_graph(names)
    #     print("a", graph.edges, indices)

    #     inputs, outputs = [], []
    #     input_shapes, output_shapes = [], []
    #     for edge in graph.edges:
    #         sn, en = edge
    #         if sn not in indices:
    #             names, shapes = connection_by_edge(graph, edge)
    #             inputs += names
    #             input_shapes += shapes

    #         if en not in indices:
    #             names, shapes = connection_by_edge(graph, edge)
    #             outputs += names
    #             output_shapes += shapes

    #     outputs = list(set(outputs))
    #     output_shapes = list(set(output_shapes))

    #     # Root and Reaf
    #     if len(inputs) == 0:
    #         for node in block[0]:
    #             inputs += node["inputs"]
    #             input_shapes += node["input_shapes"]

    #     if len(outputs) == 0:
    #         search_list = [3, 2, 1, 0]
    #         for i in search_list:
    #             if True if len(block[i]) > 0 else False:
    #                 for node in block[i]:
    #                     outputs += node["outputs"]
    #                     output_shapes += node["output_shapes"]

    #     return inputs, outputs, input_shapes, output_shapes

    def _get_block_inout(self, layers):
        empty = True
        if len(layers[0]) > 0:
            output_names = layers[0][0]["outputs"]
            output_shapes = layers[0][0]["output_shapes"]
            input_names = layers[0][0]["inputs"]
            input_shapes = layers[0][0]["input_shapes"]
            empty = False

        if len(layers[1]) > 0:
            output_names = layers[1][0]["outputs"]
            output_shapes = layers[1][0]["output_shapes"]
            if empty:
                input_names = layers[1][0]["inputs"]
                input_shapes = layers[1][0]["input_shapes"]
                empty = False

        if len(layers[2]) > 0:
            output_names = layers[2][0]["outputs"]
            output_shapes = layers[2][0]["output_shapes"]
            if empty:
                input_names = layers[2][0]["inputs"]
                input_shapes = layers[2][0]["input_shapes"]
                empty = False
            else:
                input_names += layers[2][0]["inputs"][1:2]
                input_shapes += layers[2][0]["input_shapes"][1:2]

        if len(layers[3]) > 0:
            output_names = layers[3][0]["outputs"]
            output_shapes = layers[3][0]["output_shapes"]
            if empty:
                input_names = layers[3][0]["inputs"]
                input_shapes = layers[3][0]["input_shapes"]
                empty = False

        org_shapes = output_shapes
        if len(layers[4]) > 0:
            output_names = layers[4][0]["outputs"]
            output_shapes = layers[4][0]["output_shapes"]
            if empty:
                input_names = layers[4][0]["inputs"]
                input_shapes = layers[4][0]["input_shapes"]
                empty = False

        assert empty == False
        return input_names, output_names, input_shapes, output_shapes, org_shapes

    def _init_edges(self):
        block_inputs, block_outputs = [], []
        block_input_shapes, block_output_shapes = [], []
        for bi in self.graph.nodes:
            node = self.graph.nodes[bi]
            # in_names, out_names, in_shapes, out_shapes = self._get_external_nodes(node["layers"])
            in_names, out_names, in_shapes, out_shapes, org_shapes = self._get_block_inout(node["layers"])
            node["inputs"] = in_names
            node["outputs"] = out_names
            node["input_shapes"] = in_shapes
            node["output_shapes"] = out_shapes
            node["org_shapes"] = org_shapes

        # Edge
        self.graph = update_edges(self.graph)
        return self

    def _merge_batchnormal(self, layer, epsilon=1e-7, dtype="float32"):
        if len(layer["params"]) == 1:
            return layer["params"][0], 0
        elif len(layer["params"]) == 2:
            return layer["params"]

        gamma, beta, mu, sigma = layer["params"]
        if "epsilon" in layer["configs"]:
            epsilon = layer["configs"]["epsilon"]
        sigma = np.sqrt(sigma + epsilon, dtype=dtype)
        weight = gamma / sigma
        bias = beta - ((gamma * mu) / sigma)
        return weight, bias

    def _merge_mac(self, bias, layers, dtype="float32"):
        weight = np.array([1.0], dtype=dtype)
        for layer in layers:
            _weight, _bias = self._merge_batchnormal(layer)
            weight = _weight * weight
            bias = _weight * bias + _bias

        return weight, bias


    def _convert_parameter(self):
        for bi in self.graph.nodes:
            params = []
            node = self.graph.nodes[bi]
            # Kernel
            if len(node["layers"][0]) > 0:
                layer = node["layers"][0][0]
                if len(layer["params"]) == 0: # Dummy
                    bias = np.array([0.])
                    params.append(identical_kernel(node["input_shapes"][0][-1]))
                else:
                    kernel = layer["params"][0]
                    bias = layer["params"][1] if len(layer["params"]) > 1 else np.array([0.])
                    params.append(kernel)

            else: # Dummy conv op
                # bias = identical_bias(layer["input_shapes"][0][-1])
                # params.append(identical_kernel(layer["output_shapes"][0][-1]))
                params.append(identical_kernel(node["input_shapes"][0][-1]))
                bias = identical_bias()

            # Weight & Bias
            if len(node["layers"][1]) > 0:
                layers = node["layers"][1]
                weight, bias = self._merge_mac(bias, layers)
                params.append(weight)
                params.append(bias)
            else: # Dummy
                # params.append(identical_weight(layer["output_shapes"][0][-1]))
                # params.append(identical_bias(layer["output_shapes"][0][-1]))
                params.append(identical_weight(node["output_shapes"][0][-1]))
                params.append(bias + identical_bias(node["output_shapes"][0][-1]))

            if len(node["layers"][2]) > 0:
                # layer = node["layers"][2][0]
                residual = True
            else:
                residual = False

            if len(node["layers"][3]) > 0:
                layer = node["layers"][3][0]
                activation = layer["configs"]["alpha"] if "alpha" in layer["configs"] else 0.0
            else:
                activation = None

            node["params"] = params
            node["residual"] = residual
            node["activation"] = activation

        return self

    def _assign_block_fn(self):
        for bi in self.graph.nodes:
            node = self.graph.nodes[bi]
            if node["block_function"] not in self.support_ops:
                node["block_function"] = "IdenticalConv"

        return self

    def _init_channel_index(self):
        for bi in self.graph.nodes:
            node = self.graph.nodes[bi]
            assert node["channel_index"] is None
            node["channel_index"] = [
                0,
                node["input_shapes"][0][-1],
                node["input_shapes"][0][-1],
            ]

        return self

    # def _batch_extension(self, lock):
    #     m_b = None
    #     for bi in self.graph.nodes:
    #         node = self.graph.nodes[bi]
    #         if node["block_function"] in self.lock_ops:
    #             node["input_shapes"] = self._tuple2list(node["input_shapes"])
    #             node["output_shapes"] = self._tuple2list(node["output_shapes"])
    #             node["org_shapes"] = self._tuple2list(node["output_shapes"])
    #             if m_b is None:
    #                 if len(node["input_shapes"][0]) != len(node["output_shapes"][0]):
    #                     if lock == "h":
    #                         m_b = node["input_shapes"][0][1]
    #                     elif lock == "w":
    #                         m_b = node["input_shapes"][0][2]
    #                     node["output_shapes"][0][0] *= m_b
    #                     node["org_shapes"][0][0] *= m_b
    #             else:
    #                 node["input_shapes"][0][0] *= m_b
    #                 node["output_shapes"][0][0] *= m_b
    #                 node["org_shapes"][0][0] *= m_b

    #     return self

    #################################
    ####### External - Methods
    #################################
    def create_block_model(self):
        # Create New DiGraph: Layers -> Blocks
        self._init_nodes()
        # self._merge(): TODO
        self._init_edges()
        self._convert_parameter()
        self._assign_block_fn()
        self._init_channel_index()
        # if lock is not None:
        #     self._batch_extension(lock)

        return self

    def merge_block_model(self):
        self.graph = merge_block(self.graph)
        return self

    def translate_logit_order(self, order=None):
        self.graph = reorder_logit(self.graph, order=order)
        return self

    def translate_concat_to_conv(self):
        self.graph = concat_to_conv(self.graph)
        return self

    def translate_gap_to_conv(self, mode):
        if mode == "stage":
            self.graph = gap_to_conv(self.graph)
        else:
            self.graph = gap_to_serial_conv(self.graph)
        return self

    def translate_fc_to_conv(self):
        self.graph = fc_to_conv(self.graph)
        return self

    def translate_split_conv(self, buffer_size=4096, multiple=8):
        self.graph = split_conv(self.graph, buffer_size=buffer_size, multiple=multiple)
        return self

    def translate_group_conv(self):
        self.graph = group_conv(self.graph)
        return self

    def translate_mp_to_conv(self, buffer_size=4096, multiple=8):
        self.graph = split_maxpool(self.graph, buffer_size=buffer_size, multiple=multiple)
        return self

    def translate_align_data(self, multiple=8):
        self.graph = align_data(self.graph, multiple)
        return self

    def convert_output_shapes(self):
        self.graph = reshape_outputs(self.graph)
        return self

if __name__ == '__main__':
    args = get_parser().parse_args()
    src_file = args.src_file
    dst_file = args.dst_file

    tachy_model = tload(src_file)
    tm = LAYERtoBLOCK(tachy_model)
    tm = tm.create_block_model()

    tachy_dict = tdict()
    tachy_dict['tachy_model'] = tm
    tsave(dst_file, tachy_dict)
    # cfg = tm.get_param_config()
    # print(json.dumps(cfg, indent=4))

