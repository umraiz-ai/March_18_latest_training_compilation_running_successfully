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
sys.path.append('TACHY-Compiler/compiler/src')

import argparse
import numpy as np
# import pickle

import networkx as nx
import onnx

from graph import convert_param, optimize_param, expand_param, convert_shape, update_config
from editor import cascade_concat

from tachy_format import *
from convert_onnx2tachy import ONNXtoTACHY

import ddesigner_api.numpy.xwn.functions as xwnf


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

    parser.add_argument('input_shape', type=int, nargs='+',
        help='Input shape (Batch Size, Height, Width, Channel)', 
        default=None)

    parser.add_argument('--default_pad_order', type=str,
        help='Order of default pad',
        default="down_right")

    parser.add_argument('--default_pad_mode', type=str,
        help='Mode of default pad',
        default="dynamic")

    parser.add_argument('--locked_dim', type=str,
        help='Locked dimension',
        default=None)

    parser.add_argument('--init_op_dim', type=int,
        help='Initialize operation dimension',
        default=2)

    return parser


class TachyModelONNX(ONNXtoTACHY):
    def __init__(self, model, locked_dim=None):
        super(TachyModelONNX, self).__init__(model)
        self.graph = model
        self.shp_cfg = {
            'Reshape': [
                {
                    'ORDER': ['BCHW', 'BHWC'],
                },
            ],
        }
        self.cvt_cfg = {
            # 'Conv1': [
            #     {
            #         'ORDER': ['OIW', 'WIO'],
            #         'EXPAND': [1],
            #     }, # weight
            #     {
            #     }, # bias
            # ],
            # 'Conv2': [
            #     {
            #         'ORDER': ['OIHW', 'HWIO'],
            #     }, # weight
            #     {
            #     }, # bias
            # ],
            'Conv': [
                {
                    'ORDER': ['OIHW', 'HWIO'],
                }, # weight
                {
                }, # bias
            ],
            # 'ConvTranspose1': [
            #     {
            #         'REVERSE': [2],
            #         'ORDER': ['IOW', 'WIO'],
            #         'EXPAND': [1],
            #     }, # weight
            #     {
            #     }, # bias
            # ],
            # 'ConvTranspose2': [
            #     {
            #         'REVERSE': [2, 3],
            #         'ORDER': ['IOHW', 'HWIO'],
            #     }, # weight
            #     {
            #     }, # bias
            # ],
            'ConvTranspose': [
                {
                    'REVERSE': [2, 3],
                    'ORDER': ['IOHW', 'HWIO'],
                }, # weight
                {
                }, # bias
            ],
            'Gemm': [
                {
                    'ORDER': ['OI', 'IO'],
                }, # weight
                {
                }, # bias
            ],
            'MatMul': [
                {
                    'ORDER': ['IO', 'IO'],
                }, # weight
                {
                }, # bias
            ],
            'AddP': [
                {
                    'ORDER': ['BCHW', 'BHWC'],
                }, # weight
                {
                    'ORDER': ['BCHW', 'BHWC'],
                }, # bias
            ],
            'BatchNormalization': [
                {
                    # 'EXPAND': [0, 1, 2],
                    # 'ORDER': ['BCHW', 'BHWC'],
                    'BATCHNORMAL': True,
                }, # gamma
                {
                    # 'EXPAND': [0, 1, 2],
                    # 'ORDER': ['BCHW', 'BHWC'],
                    'BATCHNORMAL': True,
                }, # beta
                {
                    # 'EXPAND': [0, 1, 2],
                    # 'ORDER': ['BCHW', 'BHWC'],
                    'BATCHNORMAL': True,
                }, # mean
                {
                    # 'EXPAND': [0, 1, 2],
                    # 'ORDER': ['BCHW', 'BHWC'],
                    'BATCHNORMAL': True,
                }, # var
            ],
        }

        self.opt_cfg = {
            # 'Conv1': [{}],
            # 'Conv2': [{}],
            # 'ConvTranspose1': [{}],
            # 'ConvTranspose2': [{}],
            'Conv': [{}],
            'ConvTranspose': [{}],
        }

        self.npu_ops = {
            0: ["Conv", "ConvTranspose", "Gemm", "MaxPool", "GlobalAveragePool", "Concat", "MatMul"],
            1: ["BatchNormalization", "PRelu", "AddP"],
            2: ["Add"],
            3: ["LeakyRelu", "Relu"],
        }
        self.locked_dim = locked_dim

    #################################
    ####### Internal
    #################################


    #################################
    ####### External - Parameters
    #################################
    def expand_parameters(self):
        self.graph = expand_param(self.graph, axis=self.locked_dim)
        return self

    def convert_parameters(self):
        self.graph = convert_param(self.graph, self.cvt_cfg)
        return self

    def convert_shapes(self):
        self.graph = convert_shape(self.graph, self.shp_cfg)
        return self

    def optimize_parameters(self, ref_graph):
        self.graph = optimize_param(self.graph, ref_graph, self.opt_cfg)
        return self


    #################################
    ####### External - Editor
    #################################
    def cascade_concatenate(self):
        self.graph = cascade_concat(self.graph) 
        return self

    def report(self):
        for bi in self.graph.nodes:
            print("[INFO]: ------------- {} Layer ------------".format(bi))
            print("[INFO]: name=", self.graph.nodes[bi]["name"])
            print("[INFO]: op_type=", self.graph.nodes[bi]["op_type"])
            print("[INFO]: inputs=", self.graph.nodes[bi]["inputs"])
            print("[INFO]: outputs=", self.graph.nodes[bi]["outputs"])
            print("[INFO]: configs=", self.graph.nodes[bi]["configs"])
            print("[INFO]: Parameter shapes=", [p.shape for p in self.graph.nodes[bi]["params"]])
            if self.graph.nodes[bi]["op_type"] == "Reshape":
                print("[INFO]: Reshape=", [p for p in self.graph.nodes[bi]["params"]])
            print("[INFO]: input_shapes=", self.graph.nodes[bi]["input_shapes"])
            print("[INFO]: output_shapes=", self.graph.nodes[bi]["output_shapes"])
            # print("[INFO]: residual=", self.graph.nodes[bi]["residual"])
            # print("[INFO]: activation=", self.graph.nodes[bi]["activation"])
            # print("[INFO]: block_function=", self.graph.nodes[bi]["block_function"])
            # print("[INFO]: channel_index=", self.graph.nodes[bi]["channel_index"])
            if "pads_dynamic" in self.graph.nodes[bi]:
                print("[INFO]: pads_dynamic=", self.graph.nodes[bi]["pads_dynamic"])
            else:
                print("[INFO]: pads_dynamic=", "Empty")

            # if "channel_residual" in self.graph.nodes[bi]:
            #     print("[INFO]: channel_residual=", self.graph.nodes[bi]["channel_residual"])
            # else:
            #     print("[INFO]: channel_residual=", "Empty")

            # for bti in self.graph.nodes[bi]["layers"]:
            #     print("[INFO]: {} NPU op=".format(bti), [layers["name"] for layers in self.graph.nodes[bi]["layers"][bti]])


            # if "op_shapes" in self.graph.nodes[bi].keys():
            #     print("[INFO]: Operation Shpaes=", self.graph.nodes[bi]["op_shapes"])

        print(self.graph)
        return self


if __name__ == '__main__':
    args = get_parser().parse_args()
    src_file = args.src_file
    dst_file = args.dst_file
    input_shape = args.input_shape
    default_pad_order = args.default_pad_order
    default_pad_mode = args.default_pad_mode
    locked_dim = args.locked_dim
    init_op_dim = args.init_op_dim

    tm = TachyModelONNX(onnx.load(src_file), locked_dim)
    tm = tm.shrink_graph(init_op_dim=init_op_dim, locked_dim=locked_dim)
    tm = tm.expand_parameters()
    tm = tm.convert_parameters()
    tm = tm.convert_shapes()
    tm = tm.cascade_concatenate()
    if input_shape is not None:
        tm.graph = update_config(
            tm.graph, input_shape,
            default_pad_order,
            default_pad_mode,
        )
    # tm.report()

    # from serialize import dumps, loads
    tachy_dict = tdict()
    tachy_dict['tachy_model'] = tm
    tsave(dst_file, tachy_dict)
    print("[INFO]: {} saved.".format(dst_file))

    # cfg = tm.get_param_config()
    # print(json.dumps(cfg, indent=4))

