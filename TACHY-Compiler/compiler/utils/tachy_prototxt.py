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
import json
import numpy as np

from tachy_format import *
from format_prototxt import *
from tachy_model import TachyModelONNX
from tachy_block import TachyBlock
# from tachy_engine import TachyEngine
from graph import build
from editor import cascade_concat


_DEBUG = False


def get_parser():
    """
    Get the parser.
    :return: parser
    """

    parser = argparse.ArgumentParser(description='Generate Prototxt from TACHY-Format')
    parser.add_argument('src_file', type=str,
                        help='Source TACHY file',
                        default=None)

    parser.add_argument('dst_file', type=str,
                        help='Destination directory path',
                        default='.')

    return parser


class TachyPrototxt:
    def __init__(
        self, 
        model, 
        param_dtype='float32', 
        op_dtype='float32', 
    ):
        self.model = model.graph
        self.p_dtype = param_dtype
        self.o_dtype = op_dtype

    #################################
    ####### Internal
    #################################
    def _get_head_lines(self):
        return []
    
    def _get_body_lines(self, defs:list):
        lines = []
        for _def in defs:
            lines += _def
            lines.append("\n")
        return lines
    
    def _get_tail_lines(self):
        return [
            "\n",
        ]

    def _get_prototxt_layer(self, node:dict):
        if node['op_type'] == 'Conv' or node['op_type'] == 'ConvTranspose' :
            info_dict = {
                'name'          : '{}'.format(node['name']),
                'tops'          : node['outputs'],
                'bottoms'       : node['inputs'],
                'shape'         : node['params'][0].shape,
                'bias'          : True if len(node['params']) == 2 else False,
                'kernel_size'   : node['configs']['kernel_shape'],
                'strides'       : node['configs']['strides'],
                'pads'          : node['configs']['pads'] if 'pads' in node['configs'] else [],
            }
            lines = get_conv_lines(info_dict)

        elif node['op_type'] == 'MaxPool':
            info_dict = {
                'name'          : '{}'.format(node['name']),
                'tops'          : node['outputs'],
                'bottoms'       : node['inputs'],
                'kernel_size'   : node['configs']['kernel_shape'],
                'strides'       : node['configs']['strides'],
                'pads'          : node['configs']['pads'],
            }
            lines = get_pooling_lines(info_dict)

        elif node['op_type'] == 'Gemm' or node['op_type'] == 'InnerProduct' or node['op_type'] == 'MatMul':
            info_dict = {
                'name'          : '{}'.format(node['name']),
                'tops'          : node['outputs'],
                'bottoms'       : node['inputs'],
                'shape'         : node['params'][0].shape,
                'bias'          : True if len(node['params']) == 2 else False,
            }
            lines = get_fc_lines(info_dict)

        elif node['op_type'] == 'AddP': 
            info_dict = {
                'name'          : '{}'.format(node['name']),
                'tops'          : node['outputs'],
                'shape'         : node['params'][0].shape,
                'bottoms'       : node['inputs'],
            }
            lines = get_add_lines(info_dict)

        elif node['op_type'] == 'Relu':
            info_dict = {
                'name'          : '{}'.format(node['name']),
                'tops'          : node['outputs'],
                'bottoms'       : node['inputs'],
                'alpha'         : 0.0,
            }
            lines = get_act_lines(info_dict)

        elif node['op_type'] == 'LeakyRelu':
            info_dict = {
                'name'          : '{}'.format(node['name']),
                'tops'          : node['outputs'],
                'bottoms'       : node['inputs'],
                'alpha'         : node['configs']['alpha'],
            }
            lines = get_act_lines(info_dict)

        elif node['op_type'] == 'Add': 
            info_dict = {
                'name'          : '{}'.format(node['name']),
                'tops'          : node['outputs'],
                'bottoms'       : node['inputs'],
            }
            lines = get_add_lines(info_dict)

        elif node['op_type'] == 'Concat': 
            info_dict = {
                'name'          : '{}'.format(node['name']),
                'tops'          : node['outputs'],
                'bottoms'       : node['inputs'],
            }
            lines = get_concat_lines(info_dict)

        elif node['op_type'] == 'GlobalAveragePool': 
            info_dict = {
                'name'          : '{}'.format(node['name']),
                'tops'          : node['outputs'],
                'bottoms'       : node['inputs'],
            }
            lines = get_gap_lines(info_dict)

        elif node['op_type'] == 'BatchNormalization':
            info_dict = {
                'name'          : '{}'.format(node['name']),
                'tops'          : node['outputs'],
                'bottoms'       : node['inputs'],
            }
            lines = get_bn_lines(info_dict)

        elif node['op_type'] == 'Reshape':
            info_dict = {
                'name'          : '{}'.format(node['name']),
                'tops'          : node['outputs'],
                'bottoms'       : node['inputs'],
                'shape'         : node['params'][0],
            }
            lines = get_reshape_lines(info_dict)

        elif node['op_type'] == 'BS3':
            info_dict = {
                'name'          : '{}'.format(node['name']),
                'tops'          : node['outputs'],
                'bottoms'       : node['inputs'],
            }
            lines = get_bs3_lines(info_dict)

        else:
            print('[WARN]: Tachy Prototxt {} not supported.'.format(node['op_type']))
            lines = []

        return lines

    def _get_body_components(self, model_name='default'):
        '''
        '''
        head = [
            "name: \"{}\"\n".format(model_name),
            "\n",
            "layer {\n",
            insert_taps("name: \"{}\"\n".format('data'), n=1),
            insert_taps("type: \"{}\"\n".format('Data'), n=1),
            "\n",
            insert_taps("top: \"{}\"\n".format('input'), n=1),
            "\n",
            insert_taps("include {\n", n=1),
            insert_taps("phase: {}\n".format('TRAIN'), n=2),
            insert_taps("}\n", n=1),
            "}\n",
            "\n",
        ]
        tail = []
    
        body = []
        for _, layer in self.layers.items():
            lines = self._get_prototxt_layer(layer.configs)
            body += lines
    
        return head + body + tail


    #################################
    ####### External - Methods
    #################################
    def generate_prototxt(self, dst_file):
        defs = []
        defs.append(self._get_body_components())

        head = self._get_head_lines()
        body = self._get_body_lines(defs)
        tail = self._get_tail_lines()
        whole = head + body + tail

        # Save
        with open(dst_file, "w") as f:
            for line in whole:
                f.write(line)

        print("[INFO] {} saved.".format(dst_file))
        return True

    def modify_and_build(self):
        # self.model = cascade_concat(self.model)
        self.inputs, self.layers, self.outputs = build(self.model, self.p_dtype, self.o_dtype)
        return self
        
    # def convert_to_block(self):
    #     # self.model = self._create_block_model()
    #     self.model = self._create_block_model()
    #     self.inputs, self.layers, self.outputs = self._build()
    #     return self
        

if __name__ == "__main__":
    args = get_parser().parse_args()
    src_file = args.src_file
    dst_file = args.dst_file

    key = "tachy_model"
    tachy_model = tload(src_file)[key]
    # print(tachy_model.graph.nodes[128])
    tp = TachyPrototxt(
        tachy_model, 
        param_dtype="float32", 
        op_dtype="float32",
    )
    tp = tp.modify_and_build()
    # tp = tp.convert_to_block()
    ret = tp.generate_prototxt(dst_file)
