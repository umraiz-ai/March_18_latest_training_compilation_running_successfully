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
import commentjson

from tachy_format import *
from tachy_block import TachyBlock

def get_parser():
    """
    Get the parser.
    :return: parser
    """

    parser = argparse.ArgumentParser(description='Generate Tachy-Runtime Network Model')
    parser.add_argument('src_file', type=str,
        help='Source TACHY file',
        default=None)

    parser.add_argument('dst_dir', type=str,
        help='Destination directory path',
        default='.')

    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    src_file = args.src_file
    dst_dir = args.dst_dir

    key = "tachy_model"
    tachy_model = tload(src_file)[key]

    input_shapes = tachy_model.blocks_config["input_shape"]
    input_inverse = tachy_model.blocks_config["use_input_inv"]

    tachy_model.instruction.tofile("{}/inst.bin".format(dst_dir))
    tachy_model.parameter.tofile("{}/param.bin".format(dst_dir))
    with open("{}/inst.json".format(dst_dir), "w") as f:
        commentjson.dump(tachy_model.blocks_config, f, indent=4)

    fname = ""
    for (h,w,d) in input_shapes:
        if len(fname) > 0:
            fname += "_"
        fname += "{}x{}x{}".format(h,w,d)

    fname += "_"
    fname += "inv-t" if input_inverse else "inv-f"

    tachy_model.runtime_model.tofile("{}/model_{}.tachyrt".format(dst_dir,fname))

