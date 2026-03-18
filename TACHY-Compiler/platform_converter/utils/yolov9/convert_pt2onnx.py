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

import torch
# from torchinfo import summary
import onnx
from onnx import shape_inference

os.environ['CUDA_VISIBLE_DEVICES'] = "0"



def get_parser():
    """
    Get the parser.
    :return: parser
    """

    parser = argparse.ArgumentParser(description='Transfer Parameters for Tensorflow-Keras.')
    parser.add_argument('SRC_PATH', type=str, 
        help='Pre-trained Source Path', default=None)

    parser.add_argument('DST_PATH', type=str, 
        help='Destination Path for Evaluation', default=None)

    parser.add_argument('INPUT_SHAPE', type=int, nargs='+',
        help='Destination Path for Evaluation', default=None)

    parser.add_argument('--FP16', dest='FP16', action='store_true',
        help='Float Point 16, default=False.')

    return parser

def main(args):
    src_path, dst_path, input_shape, fp16 = args

    # make directory
    try:
        os.mkdirs(dst_path)
    except:
        ...

    # Model
    device = torch.device('cpu')
    net = torch.load(src_path).to(device)
    # net = torch.jit.load(src_path).cpu()
    net.eval()

    # Save as model. (*.onnx)
    torch.onnx.export(
        net.float(),
        torch.empty(input_shape, dtype=torch.float32).to(device),
        dst_path, 
        export_params=True,
        # training=torch.onnx.TrainingMode.TRAINING,
        opset_version=16,
        do_constant_folding=True,
        # input_names = ['input'],
        input_names = ['images'],
        output_names = ['output'],
        # dynamic_axes = {'input':{0:'batch_size'}, 'output':{0:'batch_size'}},
    )

    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(dst_path)), dst_path)

    # Float16
    if fp16:
        from onnxconverter_common import float16
        model = onnx.load(dst_path)
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, dst_path)

    print('[INFO]: {} saved.'.format(dst_path))
    print('[INFO]: Model Transfer Done')



if __name__ == '__main__':
    args = get_parser().parse_args()
    src_path = args.SRC_PATH
    dst_path = args.DST_PATH
    input_shape = args.INPUT_SHAPE
    fp16 = args.FP16

    args = [src_path, dst_path, input_shape, fp16]
    main(args)

