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

import numpy as np

from tachy_format import *
from graph import build, feed_inputs, feed_layers, get_output
from tachy_model import TachyModelONNX


_DEBUG = False

class TachyInference:
    def __init__(
        self, 
        model, 
        param_dtype:str='float32', 
        op_dtype:str='float32', 
        default_pad_order:str="down_right", 
        dump_dir:str=None, 
    ):
        self.model = model
        self.inputs = None
        self.layers = None
        self.outputs = None
        self.param_dtype = param_dtype
        self.op_dtype = op_dtype
        self.dump_dir = dump_dir
        self.default_pad_order = default_pad_order

        self.build()

    #################################
    ####### Internal - Methods
    #################################
    #################################
    ####### External - Methods
    #################################
    def build(self):
        self.inputs, self.layers, self.outputs = build(
            self.model.graph, 
            param_dtype=self.param_dtype,
            op_dtype=self.op_dtype,
            default_pad_order=self.default_pad_order,
            dump_dir=self.dump_dir,
        )
        return self

    def get_output(self, outputs, crop=False):
        o = get_output(outputs, self.inputs, self.layers, crop=crop)
        return o

    def predict(self, source, verbose=False):
        ret = feed_inputs(self.inputs, source)
        ret = feed_layers(self.inputs, self.layers)
        logits = self.get_output(self.outputs, crop=True)
        return logits

    def debug(self, source, verbose=False):
        for key, layer in self.inputs.items():
            layer.forward(source)
    
        for key, layer in self.layers.items():
            inputs = self.get_output(layer.inputs)
            # for i in inputs: print(key, "i", i.shape)
            layer.forward(inputs)
            # for o in layer.o: print(key, "o", o.shape)

            # if key == 'Conv_8':
            # if key == 'Conv_16':
            # if key == 'BatchNormalization_17':
            # if key == 'Relu_18':
            # if key == 'MaxPool_19':
            # if key == 'Conv_20':
            # if key == 'Conv_25':
            # if key == 'Relu_27':
            # if key == 'Relu_49':
            # if key == 'Relu_111':
            # if key == 'Relu_142':

            if key == 'LeakyRelu_67':
            # if key == 'Conv_1':
                # i = np.transpose(layer.i[0][:, 0, ...],(0,2,1))
                # i = layer.i[0]
                # print(i, i.shape)
                # o = np.transpose(layer.o[0][:, 0, ...],(0,2,1))
                o = layer.o[0]
                print(o, o.shape)

            # if key == 'Block_26':
            #     print(key, layer.o[0].shape, layer.o[0])
            #     # print(key, layer.i[0].shape, layer.i[0])
    
        logits = self.get_output(self.outputs, crop=True)
        return logits
    

if __name__ == '__main__':
    model_file = '/mnt/TACHY-Station/arrhythmia_classification_resnet/TACHY-S100/SYAI_arrhyhmia-20230824-ResNet_1D/evals.tachy' 
    src_file = '/mnt/TACHY-Station/arrhythmia_classification_resnet/samples/inputs.npy' 
    param_dtype = 'float32'
    op_dtype = 'float32'


    key = "tachy_model"
    tachy_model = tload(model_file)[key]
    inf = TachyInference(
        tachy_model, 
        param_dtype=param_dtype, 
        op_dtype=op_dtype,
    )

    inputs = np.load(src_file)
    for i, sample in enumerate(inputs):
        sample = sample.T[None, :, None, :] # (C,H) -> (H,C) -> (B,H,W,C)
        output = inf.predict(sample) # [(1,2)]
        # output = inf.predict_simple(sample) # [(1,2)]
        print(i, output[0].shape, output[0])

