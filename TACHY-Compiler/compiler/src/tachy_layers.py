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

import sys, os

import numpy as np
from abc import *

from functions import get_info, extend_inputs
from functions import op_convolution, op_pooling_max, op_fullyconnected, op_mac, op_relu
# np.seterr(all='warn', over='raise')
# np.seterr(all='print')

EPSILON_16 = 9.77e-04
EPSILON_32 = 1.19e-07
EPSILON_64 = 2.22e-16
EPSILON = EPSILON_16

class Layer(metaclass=ABCMeta):
    # ToDo : edit to make dump_dir be set from user
    def __init__(
        self, 
        inputs:list, 
        configs:dict=None, 
        dump_dir:str=None, 
    ):
        self.configs = configs
        self.name = configs['name']
        self.inputs = inputs
        self.o = None
        self.i = None
        self.dump_dir = dump_dir
        self.params = []
        self.pads = None
        self.o_channel_residual = 0
        self.i_channel_residual = 0
        self.default_pad_order = configs["default_pad_order"] if "default_pad_order" in configs else None
        self.default_pad_mode = configs["default_pad_mode"] if "default_pad_mode" in configs else "dynamic"
    
    def _split_config(self, config:list, dim:int=None, value:int=0):
        '''
        Support 2D Only
        '''
        # assert len(config) > 0
        config = list(config)
        dim = int(dim)
        if dim is not None:
            # if dim == 1: # 1D
            #     if len(config) > 1: # hw
            #         config[1] = value
            #     elif len(config) > 0: # h
            #         config.append(value)

            # elif dim == 2: # 2D
            #     if len(config) > 1: # hw
            #         pass
            #     elif len(config) > 0: # h
            #         config.append(config[0])

            if len(config) > 1: # hw
                pass
            elif len(config) > 0: # h
                config.append(config[0])

        return config       

    def _get_pads(self, cfg):
        '''
        Support 2D Only
        '''
        # assert len(cfg['configs']['pads']) > 0
        dim = int(cfg['configs']['op_dim'])
        if 'pads' not in cfg['configs']:
            pads = (
                (0, 0), 
                (0, 0)
            )
            pad_mode = 'VALID'
        else:
            if len(cfg['configs']['pads']) == 2: # 1D
                pads = (cfg['configs']['pads'], (0, 0))
            elif len(cfg['configs']['pads']) == 4: # 2D
                pads = (
                    (cfg['configs']['pads'][0], cfg['configs']['pads'][2]), 
                    (cfg['configs']['pads'][1], cfg['configs']['pads'][3])
                )
            pad_mode = self.default_pad_mode

        return pads, pad_mode

    ##### update config ##### 
    def get_input_shapes(self):
        return [x.shape for x in self.i]

    def get_output_shapes(self):
        return [x.shape for x in self.o]

    def get_pads_dynamic(self):
        return self.pads
    #########################


    def extend_channel(self, x:np.ndarray, residual:int) -> np.ndarray:
        if residual > 0:
            b, h, w, c = x.shape
            # ex = np.empty((b, h, w, c + residual), dtype=x.dtype)
            ex = np.zeros((b, h, w, c + residual), dtype=x.dtype)
            ex[..., :c] = x
            return ex
        else:
            return x

    def get_pads(self, cfg, transpose=False):
        if 'auto_pad' in cfg['configs']: # (NOTSET:manual, SAME_UPPER:end, SAME_LOWER:begin, VALID)
            pad_hw = ((0, 0), (0, 0))
            if cfg['configs']['auto_pad'].decode('utf-8') == 'NOTSET':
                pad_mode = 'fixed'
                # pad_order = 'down_right'
                pad_order = 'up_left'
                pad_hw, _ = self._get_pads(cfg)

            elif cfg['configs']['auto_pad'].decode('utf-8') == 'SAME_UPPER':
                pad_mode = 'SAME'
                pad_order = 'down_right'

            elif cfg['configs']['auto_pad'].decode('utf-8') == 'SAME_LOWER':
                pad_mode = 'SAME'
                pad_order = 'up_left'

            elif cfg['configs']['auto_pad'].decode('utf-8') == 'VALID':
                pad_mode = 'VALID'
                # pad_order = 'down_right'
                pad_order = 'up_left'

            else:
                print('[ERROR]: auto_pad option is invalid. {}'.format(cfg['configs']['auto_pad'].decode('utf-8')))
                exit(0)

        else: # Default UPPER
            # pad_mode = self.default_pad_mode
            pad_hw, pad_mode = self._get_pads(cfg)
            pad_order = self._get_pad_order(pad_hw)
            if pad_order is None:
                if transpose:
                    pad_order = self.get_pad_order_inv(self.default_pad_order)
                else:
                    pad_order = self.default_pad_order

        return pad_mode, pad_hw, pad_order

    def get_pad_order_inv(self, order):
        h_ord, w_ord = order.split("_")
        # Height order
        if h_ord == "up":
            h_ord = "down"
        else: # down
            h_ord = "up"
        # Width order
        if w_ord == "right":
            w_ord = "left"
        else: # left
            w_ord = "right"

        return f"{h_ord}_{w_ord}"
        
    def _get_pad_order(self, pads):
        result = 0
        assert len(pads) == 2
        if pads[0][0] == pads[0][1]:
            mode_h = None
        elif pads[0][0] > pads[0][1]:
            mode_h = "up"
        elif pads[0][0] < pads[0][1]:
            mode_h = "down"

        if pads[1][0] == pads[1][1]:
            mode_w = None
        elif pads[1][0] > pads[1][1]:
            mode_w = "left"
        elif pads[1][0] < pads[1][1]:
            mode_w = "right"

        # Pad order
        if mode_h is None and mode_w is None:
            return None

        elif mode_h is None and mode_w is not None:
            mode_h = "up" if mode_w == "left" else "down"

        elif mode_h is not None and mode_w is None:
            mode_w = "left" if mode_h == "up" else "right"

        return "{}_{}".format(mode_h, mode_w)


    #######################################
    ##### External
    #######################################
    def get_strides(self, cfg):
        assert len(cfg['configs']['strides']) > 0
        strides = self._split_config(
            cfg['configs']['strides'], cfg['configs']['op_dim'],
            value=1
        )
        return strides

    def get_kernel_shape(self, cfg):
        assert len(cfg['configs']['kernel_shape']) > 0
        ksizes = self._split_config(
            cfg['configs']['kernel_shape'], cfg['configs']['op_dim'], # TODO: op_dim
            value=1
        )
        return ksizes

    def build_op(self, h, w, fh, fw, pad_order='down_right'):
        # if self.pad_mode is None:
        if self.pad_mode == "fixed":
            _, out_h, out_w = get_info(
                h + self.pad_hw[0][0] + self.pad_hw[0][1], 
                w + self.pad_hw[1][0] + self.pad_hw[1][1], 
                fh, fw, 
                'VALID', 
                self.stride_h, self.stride_w,
                pad_order=pad_order,
            )
            pads = self.pad_hw

        elif self.pad_mode == "dynamic":
            pads, out_h, out_w = get_info(
                h, w, 
                fh, fw, 
                "SAME",
                self.stride_h, self.stride_w,
                pad_order=pad_order,
            )
        else:
            pads, out_h, out_w = get_info(
                h, w, 
                fh, fw, 
                self.pad_mode,
                self.stride_h, self.stride_w,
                pad_order=pad_order,
            )
        return pads, out_h, out_w

    def convert_dtype(self, x_list:list, param_dtype:str=None, op_dtype:str=None):
        result = []
        for x in x_list:
            if param_dtype is not None:
                x = x.astype(param_dtype)
            if op_dtype is not None:
                x = x.astype(op_dtype)
            result.append(x)

        return result
        

    def get(self, crop):
        o = self.o
        if crop:
            if self.o_channel_residual > 0:
                o = [self.o[0][..., :-self.o_channel_residual]]
        return o

    def dump(self, dtype=np.float16) :
        print('[INFO]: {} dumped.'.format(self.name))
        if os.path.exists(os.path.join(os.getcwd(), self.dump_dir)) is not True:
            print('Failed to dump in/out. Check dump directory path')
            return

        for idx, l in enumerate(self.i) :
            l = l.astype(dtype)
            fname = '{}_input_{}'.format(self.name, idx)
            if l.ndim == 4:
                # l.transpose((0,1,3,2)).byteswap().tofile(os.path.join(os.getcwd(), self.dump_dir, '{}.bin'.format(fname)))
                l.byteswap().tofile(os.path.join(os.getcwd(), self.dump_dir, '{}.bin'.format(fname)))
            elif l.ndim == 2:
                # l.transpose((1,0)).byteswap().tofile(os.path.join(os.getcwd(), self.dump_dir, '{}.bin'.format(fname)))
                l.byteswap().tofile(os.path.join(os.getcwd(), self.dump_dir, '{}.bin'.format(fname)))
            np.save(os.path.join(os.getcwd(), self.dump_dir, '{}.npy'.format(fname)), l)

        for idx, l in enumerate(self.o) :
            l = l.astype(dtype)
            fname = '{}_output_{}'.format(self.name, idx)
            if l.ndim == 4:
                # l.transpose((0,1,3,2)).byteswap().tofile(os.path.join(os.getcwd(), self.dump_dir, '{}.bin'.format(fname)))
                l.byteswap().tofile(os.path.join(os.getcwd(), self.dump_dir, '{}.bin'.format(fname)))
            elif l.ndim == 2:
                # l.transpose((1,0)).byteswap().tofile(os.path.join(os.getcwd(), self.dump_dir, '{}.bin'.format(fname)))
                l.byteswap().tofile(os.path.join(os.getcwd(), self.dump_dir, '{}.bin'.format(fname)))
            np.save(os.path.join(os.getcwd(), self.dump_dir, '{}.npy'.format(fname)), l)
    
    @abstractmethod
    def forward(self):
        pass
    
    
######################
### Layers
######################
class Input(Layer):
    def __init__(
        self, 
        name:str, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(Input, self).__init__([], configs={'name':name}, dump_dir=dump_dir)
        self.op_dtype = op_dtype

    def forward(self, x):
        x = x if type(x) is list else [x]
        x = self.convert_dtype(x, op_dtype=self.op_dtype)
        self.o = x
        return self

### Convolution Opertations ###
class Conv(Layer):
    def __init__(
        self, 
        cfg:dict,
        inputs:list, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(Conv, self).__init__(
            inputs, 
            configs=cfg,
            dump_dir=dump_dir
        )

        # key = 'params_convert'
        key = 'params'
        assert len(cfg[key]) > 0
        self.params = cfg[key]

        w = cfg[key][0]
        b = cfg[key][1] if len(cfg[key]) > 1 else np.array([0.0])
        self.w, self.b = self.convert_dtype([w, b], param_dtype, op_dtype)

        self.stride_h, self.stride_w = self.get_strides(cfg)
        self.pad_mode, self.pad_hw, self.pad_order = self.get_pads(cfg)

        self.op_dtype = op_dtype
        self.group = cfg["configs"]["group"] if "group" in cfg["configs"] else 1

    def forward(self, inputs):
        x = inputs[0]

        fh, fw, fi, fo = self.w.shape
        n, h, w, c = x.shape

        self.pads, out_h, out_w = self.build_op(
            h, w, fh, fw, 
            pad_order=self.pad_order
        )

        out = []
        gi = c // self.group
        go = fo // self.group
        for i in range(self.group):
            if self.b.size != fo: 
                _b = self.b
            else:
                _b = self.b[..., go*i:go*(i+1)]

            o = op_convolution(
                x[..., gi*i:gi*(i+1)], 
                self.w[..., go*i:go*(i+1)], _b, 
                fh, fw, 
                out_h, out_w, 
                self.stride_h, self.stride_w, 
                self.pads, 
                dtype=self.op_dtype,
            )
            out.append(o)

        # Default
        self.o = [np.concatenate(out, axis=-1)]
        self.i = inputs
        
        if self.dump_dir is not None: 
            self.dump() 

        return self



class ConvTranspose(Layer):
    def __init__(
        self, 
        cfg:dict,
        inputs:list, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(ConvTranspose, self).__init__(
            inputs, 
            configs=cfg,
            dump_dir=dump_dir
        )

        # key = 'params_convert'
        key = 'params'
        assert len(cfg[key]) > 0
        self.params = cfg[key]

        w = cfg[key][0]
        b = cfg[key][1] if len(cfg[key]) > 1 else np.array([0.0])
        self.w, self.b = self.convert_dtype([w, b], param_dtype, op_dtype)

        self.stride_h, self.stride_w = self.get_strides(cfg)
        self.pad_mode, self.pad_hw, self.pad_order = self.get_pads(cfg, transpose=True)

        self.op_dtype = op_dtype
        # self.pad_mode = 'TRANSPOSE'
        # self.pad_order = 'up_left'

    def forward(self, inputs):
        x = inputs[0]

        fh, fw, fi, fo = self.w.shape
        n, h, w, c = x.shape

        osize = [h * self.stride_h, w * self.stride_w]
        x = extend_inputs(
            x, osize, (self.stride_h, self.stride_w), (fh, fw), 
            dtype=self.op_dtype
        )

        n, h, w, c = x.shape
        self.pads, out_h, out_w = get_info(
            h, w, 
            fh, fw, 
            self.pad_mode,
            1, 1,
            self.pad_order
        )

        out = op_convolution(
            x, 
            self.w, self.b,
            fh, fw, 
            out_h, out_w, 
            1, 1, 
            self.pads, 
            dtype=self.op_dtype,
        )

        # Default
        self.o = [out]
        self.i = inputs
        
        if self.dump_dir is not None: 
            self.dump() 

        return self


class MaxPool(Layer):
    def __init__(
        self, 
        cfg:dict,
        inputs:list, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(MaxPool, self).__init__(
            inputs, 
            configs=cfg,
            dump_dir=dump_dir
        )

        self.ksize_h, self.ksize_w = self.get_kernel_shape(cfg)
        self.stride_h, self.stride_w = self.get_strides(cfg)
        self.pad_mode, self.pad_hw, self.pad_order = self.get_pads(cfg)
        self.operation = 'MAX'
        self.op_dtype = op_dtype

    def forward(self, inputs):
        x = inputs[0]

        n, h, w, c = x.shape

        self.pads, out_h, out_w = self.build_op(
            h, w, self.ksize_h, self.ksize_w, 
            pad_order=self.pad_order
        )
        out = op_pooling_max(
            x, 
            self.ksize_h, self.ksize_w, 
            out_h, out_w, 
            self.stride_h, self.stride_w, 
            self.pads, 
            dtype=self.op_dtype, 
        )

        # Default
        self.o = [out]
        self.i = inputs
        
        if self.dump_dir is not None: 
            self.dump() 

        return self

class Concat(Layer):
    def __init__(
        self, 
        cfg:dict,
        inputs:list, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(Concat, self).__init__(
            inputs, 
            configs=cfg,
            dump_dir=dump_dir
        )
    
    def forward(self, inputs, axis=-1):
        out = np.concatenate(inputs, axis=axis)

        # Default
        self.o = [out]
        self.i = inputs
        
        if self.dump_dir is not None: 
            self.dump() 

        return self
        

class BatchNormalization(Layer):
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(
        self, 
        cfg:dict,
        inputs:list, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(BatchNormalization, self).__init__(
            inputs, 
            configs=cfg,
            dump_dir=dump_dir
        )
        
        key = 'params'
        assert len(cfg[key]) == 4
        self.params = cfg[key]

        gamma = cfg[key][0]
        beta = cfg[key][1]
        running_mean = cfg[key][2]
        running_var = cfg[key][3]  
        self.epsilon = cfg["configs"]["epsilon"] if "epsilon" in cfg["configs"] else EPSILON
        self.gamma, self.beta, self.running_mean, self.running_var = self.convert_dtype(
            [gamma, beta, running_mean, np.sqrt(running_var + self.epsilon, dtype="float32")], 
            param_dtype, op_dtype
        )
        self.merged_op = False
         
        # Not use variables
        self.momentum = None
        self.input_shape = None 
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dGAMMA = None
        self.dBETA = None

    def forward(self, inputs, train_flg=False):
        x = inputs[0]

        self.input_shape = x.shape
        out = self.__forward(x, train_flg)
        out = out.reshape(*self.input_shape) # NCHW -> NHWC

        # Default
        self.o = [out]
        self.i = inputs
        
        if self.dump_dir is not None: 
            self.dump() 

        return self


    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D, dtype=self.dtype)
            self.running_var = np.zeros(D, dtype=self.dtype)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + self.epsilon)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            if self.merged_op:
                w = self.gamma / self.running_var
                b = self.beta - ((self.gamma * self.running_mean) / self.running_var)
                xn = w * x + b
            else:
                xn = self.gamma * (x - self.running_mean) / (self.running_var) + self.beta
            
        out = xn
        return out


class Relu(Layer):

    def __init__(
        self, 
        cfg:dict,
        inputs:list, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(Relu, self).__init__(
            inputs, 
            configs=cfg,
            dump_dir=dump_dir
        )
        self.mask = None

    def forward(self, inputs):
        x = inputs[0]
        out = op_relu(x)

        # Default
        self.o = [out]
        self.i = inputs
        
        if self.dump_dir is not None: 
            self.dump() 

        return self

class LeakyRelu(Layer):

    def __init__(
        self, 
        cfg:dict,
        inputs:list, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(LeakyRelu, self).__init__(
            inputs, 
            configs=cfg,
            dump_dir=dump_dir
        )
        self.alpha = cfg['configs']['alpha']

    def forward(self, inputs):
        x = inputs[0]
        out = np.where(x >= 0, x, x * self.alpha)

        # Default
        self.o = [out]
        self.i = inputs
        
        if self.dump_dir is not None: 
            self.dump() 

        return self


class Add(Layer):

    def __init__(
        self, 
        cfg:dict,
        inputs:list, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(Add, self).__init__(
            inputs, 
            configs=cfg,
            dump_dir=dump_dir
        )

    def forward(self, inputs):
        #todo: matching dimension
        x, y = inputs
        out = x + y

        # Default
        self.o = [out]
        self.i = inputs
        
        if self.dump_dir is not None: 
            self.dump() 

        return self

class AddP(Layer):

    def __init__(
        self, 
        cfg:dict,
        inputs:list, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(AddP, self).__init__(
            inputs, 
            configs=cfg,
            dump_dir=dump_dir
        )

        key = 'params'
        assert len(cfg[key]) > 0
        self.params = cfg[key]
        # w = cfg[key][0]
        b = cfg[key][1]
        self.b = self.convert_dtype([b,], param_dtype, op_dtype)[0]

    def forward(self, inputs):
        #todo: matching dimension
        x = inputs[0]
        out = x + self.b

        # Default
        self.o = [out]
        self.i = inputs
        
        if self.dump_dir is not None: 
            self.dump() 

        return self



class GlobalAveragePool(Layer):
    def __init__(
        self, 
        cfg:dict,
        inputs:list, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(GlobalAveragePool, self).__init__(
            inputs, 
            configs=cfg,
            dump_dir=dump_dir
        )
        self.locked_dim = cfg["configs"]["locked_dim"] if cfg["configs"]["op_dim"] == 1 else None

    def forward(self, inputs):
        x = inputs[0]
        n, h, w, c = x.shape

        if self.locked_dim == "h":
            out = np.mean(x, axis=(2,))[0]
        elif self.locked_dim == "w":
            out = np.mean(x, axis=(1,)) [0]
        else:
            out = np.mean(x, axis=(1,2)) 
        out = np.reshape(out, (-1, c))

        # Default
        self.o = [out]
        self.i = inputs
        
        if self.dump_dir is not None: 
            self.dump() 

        return self

class MatMul(Layer):

    def __init__(
        self, 
        cfg:dict,
        inputs:list, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(MatMul, self).__init__(
            inputs, 
            configs=cfg,
            dump_dir=dump_dir
        )

        # key = 'params_convert'
        key = 'params'
        assert len(cfg[key]) > 0
        self.params = cfg[key]

        w = cfg[key][0]
        self.w = self.convert_dtype([w,], param_dtype, op_dtype)[0]

    def forward(self, inputs):
        x = inputs[0]

        x = x.reshape(x.shape[0], -1)
        out = np.dot(x, self.w)

        # Default
        self.o = [out]
        # self.i = inputs
        self.i = [x]
        
        if self.dump_dir is not None: 
            self.dump() 

        return self


class Gemm(Layer):

    def __init__(
        self, 
        cfg:dict,
        inputs:list, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(Gemm, self).__init__(
            inputs, 
            configs=cfg,
            dump_dir=dump_dir
        )

        # key = 'params_convert'
        key = 'params'
        assert len(cfg[key]) > 0
        self.params = cfg[key]

        w = cfg[key][0]
        b = cfg[key][1] if len(cfg[key]) > 1 else np.array([0.0])
        self.w, self.b = self.convert_dtype([w, b], param_dtype, op_dtype)

    def forward(self, inputs):
        x = inputs[0]

        out = op_fullyconnected(x, self.w, self.b)

        # Default
        self.o = [out]
        self.i = inputs
        
        if self.dump_dir is not None: 
            self.dump() 

        return self

class Reshape(Layer):

    def __init__(
        self, 
        cfg:dict,
        inputs:list, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(Reshape, self).__init__(
            inputs, 
            configs=cfg,
            dump_dir=dump_dir
        )

        key = 'params'
        assert len(cfg[key]) > 0
        self.shape = cfg[key][0]

    def forward(self, inputs):
        x = inputs[0]
        out = x.reshape(self.shape)
        # Default
        self.o = [out]
        self.i = [x]
        
        if self.dump_dir is not None: 
            self.dump() 

        return self


class BS3(Layer):

    def __init__(
        self, 
        cfg:dict,
        inputs:list, 
        param_dtype:str, 
        op_dtype:str, 
        dump_dir=None
    ):
        super(BS3, self).__init__(
            inputs, 
            configs=cfg,
            dump_dir=dump_dir
        )

        key = 'params'
        assert len(cfg[key]) > 0
        self.params = cfg[key]

        k, w, b = cfg[key]
        self.kw, self.w, self.b = self.convert_dtype(
            [k, w, b], param_dtype, op_dtype
        )
        self.kb = np.array([0.0], dtype=op_dtype)

        self.ksize_h, self.ksize_w = self.get_kernel_shape(cfg)
        self.stride_h, self.stride_w = self.get_strides(cfg)

        self.op_dtype = op_dtype
        self.activation = cfg["activation"]
        self.fn = cfg["block_function"]
        self.i_channel_residual = cfg["channel_residual"][0] if "channel_residual" in cfg else 0
        self.o_channel_residual = cfg["channel_residual"][1] if "channel_residual" in cfg else 0
        
        if self.fn == "SerialConv":
            self.get_input_shapes = self.convert_input_shapes

        if self.fn == "ConvTranspose":
            self.pad_mode, self.pad_hw, self.pad_order = self.get_pads(cfg, transpose=True)
        else:
            self.pad_mode, self.pad_hw, self.pad_order = self.get_pads(cfg, transpose=False)

    def convert_input_shapes(self):
        shapes = []
        for i, x in enumerate(self.i):
            if i == 0:
                stt, end, _ = self.configs["channel_index"]
                n_channel = end - stt
                shapes.append((x.shape[0], 1, 1, n_channel))
            else:
                shapes.append(x.shape)
        return shapes


    def forward(self, inputs):
        x = inputs[0]
        y = inputs[1] if len(inputs) > 1 else self.kb

        # # Input Residual
        # x = self.extend_channel(x, self.i_channel_residual)

        if self.fn == "Conv" or self.fn == "IdenticalConv":
            stt, end, _ = self.configs["channel_index"]
            # assert end - stt == self.kw.shape[2], f"[ERROR]: {self.configs["name"]}"
            assert end - stt == self.kw.shape[2], f"[ERROR]: {self.name} {end}-{stt}, {self.kw.shape[2]}"
            x = x[..., stt:end]

            # # FC -> CONV Dim
            # if "input_shapes" in self.configs.keys():
            #     input_shape = list(self.configs["input_shapes"][0]) # ex: [1,208,112,32]
            #     input_shape[-1] = self.configs["channel_index"][1] - self.configs["channel_index"][0] # ex: [1,208,112,16]
            #     if x.shape != tuple(input_shape):
            #         x = x.reshape(input_shape)

            # inputs[0] = x
            n, h, w, c = x.shape
            fh, fw, fi, fo = self.kw.shape

            self.pads, out_h, out_w = self.build_op(
                h, w, fh, fw, 
                pad_order=self.pad_order
            )
            # print(
            #     x.shape, 
            #     self.kw.shape, self.kb.shape, 
            #     fh, fw, 
            #     out_h, out_w, 
            #     self.stride_h, self.stride_w, 
            #     self.pad_mode,
            #     self.pads, 
            #     self.op_dtype,
            # )

            x = op_convolution(
                x, 
                self.kw, self.kb, 
                fh, fw, 
                out_h, out_w, 
                self.stride_h, self.stride_w, 
                self.pads, 
                dtype=self.op_dtype,
            )
        elif self.fn == "ConvTranspose":
            stt, end, _ = self.configs["channel_index"]
            x = x[..., stt:end]

            fh, fw, fi, fo = self.kw.shape
            n, h, w, c = x.shape

            osize = [h * self.stride_h, w * self.stride_w]
            x = extend_inputs(
                x, osize, (self.stride_h, self.stride_w), (fh, fw), 
                dtype=self.op_dtype
            )

            n, h, w, c = x.shape
            self.pads, out_h, out_w = get_info(
                h, w,
                fh, fw,
                self.pad_mode,
                1, 1,
                self.pad_order
            )
            # self.pads, out_h, out_w = get_info(
            #     h, w,
            #     fh, fw,
            #     "TRANSPOSE",
            #     1, 1,
            #     "up_left"
            # )

            x = op_convolution(
                x, 
                self.kw, self.kb, 
                fh, fw, 
                out_h, out_w, 
                1, 1, 
                self.pads, 
                dtype=self.op_dtype,
            )

        elif self.fn == "MaxPool":
            n, h, w, c = x.shape
            self.pads, out_h, out_w = self.build_op(
                h, w, self.ksize_h, self.ksize_w, 
                pad_order=self.pad_order
            )
            x = op_pooling_max(
                x, 
                self.ksize_h, self.ksize_w, 
                out_h, out_w, 
                self.stride_h, self.stride_w, 
                self.pads, 
                dtype=self.op_dtype, 
            )

        elif self.fn == "Gemm" or self.fn == "MatMul":
            x = op_fullyconnected(x, self.kw, self.kb)

        elif self.fn == "SerialConv":
            stt, end, _ = self.configs["channel_index"]
            x = x.transpose(0,1,3,2).reshape(-1)[stt:end].reshape(1, 1, 1, -1) # (B,H,W,C) -> (B,H,C,W)

            n, h, w, c = x.shape
            fh, fw, fi, fo = self.kw.shape
            self.pads, out_h, out_w = self.build_op(
                h, w, fh, fw, 
                pad_order=self.pad_order
            )
            x = op_convolution(
                x, 
                self.kw, self.kb, 
                fh, fw, 
                out_h, out_w, 
                self.stride_h, self.stride_w, 
                self.pads, 
                dtype=self.op_dtype,
            )
            # x = x.transpose(0,1,3,2) # (B,H,C,W) -> (B,H,W,C)
            
        x = op_mac(x, self.w, self.b)

        assert x.size == y.size or y.size == 1
        if x.shape != y.shape and y.size > 1:
            y = np.reshape(y, x.shape)
        x = x + y

        if self.activation is not None:
            x = op_relu(x, alpha=self.activation)

        # Reshape output
        if "output_shapes" in self.configs.keys():
            output_shape = list(self.configs["output_shapes"][0]) # ex: [1,208,112,32]
            x = x.reshape(output_shape)
        out = x

        # Default
        self.o = [out]
        self.i = inputs
        
        if self.dump_dir is not None: 
            self.dump() 

        return self



if __name__ == '__main__':
    w = np.load('/media/hdd2/mk/20230224/DDesigner/work/tconv_weight.npy')
    inputs = np.load('/media/hdd2/mk/20230224/DDesigner/work/tconv_input.npy')
    cfg = {
        'name': 'tconv_test',
        # 'params_convert': [w],
        'params': [w],
        'configs': {
            'op_dim' : 2,
            'pads' : [2, 0, 2, 0],
            'auto_pad' : 'NOTSET'.encode('utf-8'),
            'strides' : [2, 2],
        },
    }
    tconv = ConvTranspose(cfg, [], 'float32', 'float32')
    print(tconv.forward([inputs]).o)


