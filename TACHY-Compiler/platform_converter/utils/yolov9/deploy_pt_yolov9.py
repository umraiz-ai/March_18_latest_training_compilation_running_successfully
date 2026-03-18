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



import os
import argparse

import numpy as np
import torch
import torch.nn as nn
# from torchinfo import summary

from ddesigner_api.pytorch.xwn import torch_nn as cnn
import ddesigner_api.numpy.xwn.optimization as xwn



def get_parser():
    """
    Get the parser.
    :return: parser
    """

    parser = argparse.ArgumentParser(description='Transfer Parameters for PyTorch.')
    parser.add_argument('SRC_PRE_PATH', type=str, 
        help='Pre-trained Source Path', default=None)
    parser.add_argument('SRC_OPT_PATH', type=str, 
        help='Optimization-trained Source Path', default=None)

    parser.add_argument('DST_EVAL_PATH', type=str, 
        help='Destination Path for Evaluation', default=None)
    parser.add_argument('DST_COMP_PATH', type=str, 
        help='Destination Path for Compile', default=None)
    parser.add_argument('--DST_TEMP_PATH', dest='DST_TEMP_PATH', type=str, 
        help='Destination Path for Raw', default=None)

    return parser

def main(args):
    src_pre_path, src_opt_path, dst_eval_path, dst_comp_path, dst_temp_path = args

    # Model
    # net_opt = torch.load(src_opt_path, map_location='cpu')['ema']       # With XWN Logic
    # net_compile = torch.load(src_pre_path, map_location='cpu')['ema']   # Without XWN Logic
    # net_verify = torch.load(src_pre_path, map_location='cpu')['ema']    # Without XWN Logic
    net_opt = torch.load(src_opt_path, map_location='cpu')['model']       # With XWN Logic
    net_compile = torch.load(src_pre_path, map_location='cpu')['model']   # Without XWN Logic
    net_verify = torch.load(src_pre_path, map_location='cpu')['model']    # Without XWN Logic
    # summary(net_opt, (128, 1, 700), depth=5)
    
    # Load parameters
    net_compile.load_state_dict(net_opt.state_dict())
    net_verify.load_state_dict(net_opt.state_dict())
    
    # Overwrite parameters
    for mod_x, mod_v in zip(net_opt.modules(), net_verify.modules()):
        if isinstance(mod_x, cnn.Conv1d):
            use_transform = mod_x.opt.use_transform
            bit = mod_x.opt.bit
            max_scale = mod_x.opt.max_scale
            use_pruning = mod_x.opt.use_pruning
            prun_weight = mod_x.opt.prun_weight
            transpose = mod_x.opt.transpose
    
            x = mod_x.weight.data.numpy()
            x = np.transpose(x, (2,1,0))
            if len(x.shape) == 3:
                opt = xwn.Optimization(
                    use_transform=use_transform,
                    bit=bit,
                    max_scale=max_scale,
                    use_pruning=use_pruning,
                    prun_weight=prun_weight,
                    transpose=transpose,
                    shape = x.shape,
                )
                x = opt.optimize(x)
                x = np.transpose(x, (2,1,0))
                mod_v.weight.data = torch.Tensor(x)
    
        elif isinstance(mod_x, cnn.Conv2d):
            use_transform = mod_x.opt.use_transform
            bit = mod_x.opt.bit
            max_scale = mod_x.opt.max_scale
            use_pruning = mod_x.opt.use_pruning
            prun_weight = mod_x.opt.prun_weight
            transpose = mod_x.opt.transpose
    
            x = mod_x.weight.data.numpy()
            x = np.transpose(x, (2,3,1,0))
            if len(x.shape) == 4:
                opt = xwn.Optimization(
                    use_transform=use_transform,
                    bit=bit,
                    max_scale=max_scale,
                    use_pruning=use_pruning,
                    prun_weight=prun_weight,
                    transpose=transpose,
                    shape = x.shape,
                )
                x = opt.optimize(x)
                x = np.transpose(x, (3,2,0,1))
                mod_v.weight.data = torch.Tensor(x)
    
    
    # Save as model. (*.pt)
    # Verification
    torch.save(net_verify, dst_eval_path)
    print('[INFO]: {} saved.'.format(dst_eval_path))
    
    # Compile
    torch.save(net_compile, dst_comp_path)
    print('[INFO]: {} saved.'.format(dst_comp_path))
    
    # Raw
    if dst_temp_path is not None:
        torch.save(net_opt, dst_temp_path)
        print('[INFO]: {} saved.'.format(dst_temp_path))
    
    print('[INFO]: Model Transfer Done')


if __name__ == '__main__':
    args = get_parser().parse_args()
    src_pre_path = args.SRC_PRE_PATH
    src_opt_path = args.SRC_OPT_PATH
    dst_eval_path = args.DST_EVAL_PATH
    dst_comp_path = args.DST_COMP_PATH
    dst_temp_path = args.DST_TEMP_PATH

    args = [src_pre_path, src_opt_path, dst_eval_path, dst_comp_path, dst_temp_path]
    main(args)

