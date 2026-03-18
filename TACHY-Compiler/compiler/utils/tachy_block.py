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
import subprocess
import commentjson

import numpy as np
import networkx as nx

from tachy_format import *
from graph import build, update_config, optimize_param, optimize_precision, invert_input_channel
from tachy_model import TachyModelONNX
from convert_layer2block import LAYERtoBLOCK
from extract import *
from convert_tachy2tachyrt import *
from tachyrt_model import tachyrt_model
from verify import *
from constraint import *
from map_address import *


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

    parser.add_argument('shape', type=int, nargs='+',
        help='Input shape (Batch Size, Height, Width, Channel)', 
        default=None)

    parser.add_argument('--default_pad_order', type=str,
        help='Order of default pad',
        default="down_right")

    parser.add_argument('--default_pad_mode', type=str,
        help='Mode of default pad',
        default="dynamic")

    parser.add_argument('--strategy_gap', type=str,
        help='Strategy of global average pooling',
        default="serial")

    parser.add_argument('--logit_order', type=int, nargs='+',
        help='Order of logits', 
        default=None)

    parser.add_argument('--work_dir', type=str,
        help='Directory path for work',
        default="work")

    parser.add_argument('--script_dir', type=str,
        help='Directory path for script',
        default="utils")

    parser.add_argument('--ref_file', type=str,
        help='Reference directory path',
        default=None)

    return parser



class TachyBlock(LAYERtoBLOCK):
    def __init__(
        self, 
        layers, 
        input_shape:list,
        default_pad_order:str="down_right", 
        default_pad_mode:str="dynamic", 
        strategy_gap:str="serial", 
        logit_order:list=None, 
        work_dir:str="work", 
        script_dir:str="utils", 
    ):
        super(TachyBlock, self).__init__(layers)
        self.input_shape = self._extend_1d(input_shape)
        self.buffer_size = 4096

        self.opt_cfg = {
            'BS3': [{}],
        }

        self.work_dir = work_dir
        self.script_dir = script_dir
        self.use_opt_precision = False
        self.use_input_inv = False
        self.multiple = 8
        self.use_scaler = [True]
        self.mode_gap = strategy_gap # serial, stage
        # self.locked_dim = "h" # serial, stage
        self.logit_order = logit_order if logit_order is not None else [] 
        self.default_pad_order = default_pad_order
        self.default_pad_mode = default_pad_mode

        if not os.path.isdir(self.work_dir):
            os.mkdir(self.work_dir)

    #################################
    ####### Internal
    #################################
    def _extend_1d(self, shape):
        assert len(shape) == 3 or len(shape) == 4

        # 1D
        if len(shape) == 3:
            b, h, c = shape
            shape = (1, h, b, c)

        return shape

    def _as_binary(
        self, 
        x, 
        dtype="float16", 
        size_list:tuple=(2, 4), 
        dtype_list:tuple=("uint8", "bool", "uint8")
    ):
        # Shape
        if x.ndim in size_list:
            if x.ndim == 4:
                h, w, i, o = x.shape
                x = x.transpose(3, 2, 0, 1).reshape(o * i, h * w)

            elif x.ndim == 2:
                i, o = x.shape
                x = x.T

        # Data Type
        if x.dtype in dtype_list:
            x = x.byteswap()
        else:
            x = x.astype(dtype).byteswap()

        return x

    def _get_reserved_bit_size(self, h, w, xwn_bit, align_bit=16, header_bit=7) -> int:
        size = int(max(w, h))
        return (align_bit - header_bit) - ((size * size * xwn_bit) % align_bit)

    def _update_binary_params(self):
        key_list = [
            "params_header",
            "params_magnitude",
            "params_sign",
            "params_scale",
        ]
        for ni in self.graph.nodes:
            node = self.graph.nodes[ni]
            # Basic
            node["params_binary"] = [self._as_binary(p) for p in node["params"]]
            # Optimized
            for k in key_list:
                if k in node: node[k] = self._as_binary(node[k])
        return self

    def _cmd_fullyconnected(self, node) -> list:
        i, o = node["params"][0].shape
        # Path
        prefix = "{}_".format(node["name"])
        exe_path = os.path.join(self.script_dir, "fc.out")
        kernel_path = os.path.join(self.work_dir, prefix + "kernel.bin")
        weight_path = os.path.join(self.work_dir, prefix + "weight.bin")
        bias_path = os.path.join(self.work_dir, prefix + "bias.bin")
        header_path = os.path.join(self.work_dir, prefix + "header.bin")

        # Save
        node["params_binary"][0].tofile(kernel_path)
        node["params_binary"][1].tofile(weight_path)
        node["params_binary"][2].tofile(bias_path)
        node["params_header"].tofile(header_path)

        # Command
        cmd = [exe_path]
        cmd += [kernel_path, bias_path, header_path, weight_path]
        cmd += [i, o]
        cmd += [str(9)]
        return cmd

    def _cmd_convolution(self, node, xwn_bit=4) -> list:
        if "params_shape" in node:
            kh, kw, i, o = node["params_shape"]
        else:
            kh, kw, i, o = node["params"][0].shape
        ks = max(kh, kw)

        # Path
        prefix = "{}_".format(node["name"])
        exe_path = os.path.join(self.script_dir, "block_4bit.out")
        kernel_path = os.path.join(self.work_dir, prefix + "kernel.bin")
        weight_path = os.path.join(self.work_dir, prefix + "weight.bin")
        bias_path = os.path.join(self.work_dir, prefix + "bias.bin")
        header_path = os.path.join(self.work_dir, prefix + "header.bin")
        magnitude_path = os.path.join(self.work_dir, prefix + "magnitude.bin")
        sign_path = os.path.join(self.work_dir, prefix + "sign.bin")
        scale_path = os.path.join(self.work_dir, prefix + "scale.bin")

        # Save
        node["params_binary"][0].tofile(kernel_path)
        node["params_binary"][1].tofile(weight_path)
        node["params_binary"][2].tofile(bias_path)
        node["params_header"].tofile(header_path)
        # if node["name"] == "Block_119":
        #     print(node["params_header"], node["params_header"].dtype, node["params_header"].shape)
        #     print(node["params_magnitude"])
        node["params_sign"].tofile(sign_path)
        node["params_magnitude"].tofile(magnitude_path)
        node["params_scale"].tofile(scale_path)

        # Command
        cmd = [exe_path]
        cmd += [weight_path, bias_path, sign_path, magnitude_path, header_path, scale_path]
        cmd += [ks, ks, i, o]
        cmd += [self._get_reserved_bit_size(kh, kw, xwn_bit), xwn_bit - 1]
        return cmd


    def _update_commands(self):
        for ni in self.graph.nodes:
            block_fn = self.graph.nodes[ni]["block_function"]
            if block_fn == "Conv" or block_fn == "ConvTranspose" or block_fn == "IdenticalConv" or block_fn == "ConvSplit" or block_fn == "SerialConv" or block_fn == "ConvTransposeSplit":
                self.graph.nodes[ni]["params_cmd"] = self._cmd_convolution(self.graph.nodes[ni])
            elif block_fn == "Gemm":
                self.graph.nodes[ni]["params_cmd"] = self._cmd_fullyconnected(self.graph.nodes[ni])
            elif block_fn == "MaxPool":
                self.graph.nodes[ni]["params_cmd"] = None
            else:
                self.graph.nodes[ni]["params_cmd"] = None

        return self

    def _extern_command_return_line(self, command) -> list:
        """
        Excute exernal sub process
        :param command: string
        :return: list of string
        """
        p = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines = p.stdout.readlines()
        result = []
        for line in lines:
            if line != '':
                result.append(line)
    
        p.wait()
        return result

    def _execute_commands(self, verbose=False):
        for ni in self.graph.nodes:
            cmd = self.graph.nodes[ni]["params_cmd"]
            if cmd is not None:
                cmd = " ".join([str(s) for s in cmd])
                if verbose: print("[INFO]: {}".format(cmd))
                retline = self._extern_command_return_line(cmd)
                self.graph.nodes[ni]["params_size"] = int(retline[0].strip()) if len(retline) > 0 else 0 
            else:
                self.graph.nodes[ni]["params_size"] = 0 

        return self

    def _add_constraint_option(self):
        for i in range(self.blocks_config["n_block"]):
            self.blocks_config["blocks"][i] = align_odd_width(self.blocks_config["blocks"][i])
            self.blocks_config["blocks"][i] = add_dummy_padding(self.blocks_config["blocks"][i])

    def _extract_block_config(self):
        self.blocks_config = {}
        self.blocks_config["blocks"] = []
        self.blocks_config["n_block"] = len(self.graph.nodes)
        param_offset = 0
        for idx, bi in enumerate(self.graph.nodes):
            cfg = {}
            ''' TODO: when change logit ordering, bi does not matches with name'''
            cfg["bi"] = int(self.graph.nodes[bi]["name"].split('_')[-1])
            cfg["index"] = idx
            cfg["operation"] = get_operation(self.graph.nodes, bi)
            cfg["input_shape"] = get_input_shape(self.graph.nodes, bi)
            cfg["output_shape"] = get_output_shape(self.graph.nodes, bi)
            cfg["kernel_shape"] = get_kernel_shape(self.graph.nodes, bi)
            cfg["stride_shape"] = get_stride_shape(self.graph.nodes, bi)
            cfg["padding_shape"] = get_padding_shape(self.graph.nodes, bi)
            cfg["residual_with"] = get_residual_with(self.graph.nodes, bi)
            cfg["input_from"] = get_input_from(self.graph, bi)
            cfg["output_tos"] = get_output_tos(self.graph, bi)
            cfg["xwn_bit"] = get_xwn_bit(self.graph.nodes, bi)
            cfg["xwn_scale"] = get_xwn_scale(self.graph.nodes, bi)
            cfg["param_size"] = get_param_size(self.graph.nodes, bi)
            cfg["is_start"] = get_is_start(self.graph.nodes, bi)
            cfg["is_logit"] = get_is_logit(self.graph, bi)
            cfg["first_depth"] = get_first_depth(self.graph.nodes, bi)
            cfg["channel_idx"] = get_channel_idx(self.graph.nodes, bi)
            cfg["channel_residual"] = get_channel_residual(self.graph.nodes, bi)
            cfg["activation"] = get_activation(self.graph.nodes, bi)
            cfg["param_offset"] = param_offset

            param_offset += get_param_size(self.graph.nodes, bi)
            self.blocks_config["blocks"].append(cfg)

    def _create_scaler(self):
        param_align = 128
        scaler = self.blocks_config["blocks"][0].copy()
        scaler["bi"] = -1
        scaler["index"] = 0
        scaler["operation"] = "Scaler"
        scaler["is_start"] = True
        scaler["input_from"] = -1
        scaler["output_tos"] = [1]
        scaler["output_shape"] = scaler["input_shape"]
        scaler["kernel_shape"] = [1,1,scaler["input_shape"][2],scaler["input_shape"][2]]
        scaler["stride_shape"] = [1,1]
        scaler["padding_shape"] = [0,0,0,0,0] # TODO: add padding
        scaler["xwn_bit"] = 1
        scaler["xwn_scale"] = 1.0
        scaler["residual_with"] = None
        scaler["activation"] = None
        scaler["channel_idx"] = [0, scaler["input_shape"][2], scaler["input_shape"][2]]
        scaler["channel_residual"] = [0,0]
        scaler["param_size"] = 4 + (4 * scaler["input_shape"][2]) + 16 # 4byte(bnw + bnb) + (4byte * in_d((header+alpha) * input depth)) + 16byte(dummy)
        scaler["param_size"] = scaler["param_size"] if scaler["param_size"] % param_align == 0 else scaler["param_size"] + (param_align - (scaler["param_size"] % param_align))
        return scaler

    def _append_scaler_block(self):
        def add_scaler_param(data):
            data = data.view(np.uint8).reshape(-1)
            param = np.fromfile('./param.bin', np.uint8).reshape(-1)
            param = np.concatenate([data, param], axis=0, dtype=np.uint8)
            param.tofile('./param.bin')

        if self.use_scaler[0]: # TODO: 
            scaler = self._create_scaler()

            for i in range(self.blocks_config["n_block"]):
                if i == 0:
                    self.blocks_config["blocks"][i]["is_start"] = False
                self.blocks_config["blocks"][i]["index"] += 1
                self.blocks_config["blocks"][i]["input_from"] += 1
                if self.blocks_config["blocks"][i]["residual_with"] is not None:
                    self.blocks_config["blocks"][i]["residual_with"] += 1
                self.blocks_config["blocks"][i]["param_offset"] += scaler["param_size"]
                self.blocks_config["blocks"][i]["output_tos"] = [x + 1 for x in self.blocks_config["blocks"][i]["output_tos"]]

            self.blocks_config["n_block"] += 1
            self.blocks_config["blocks"].insert(0, scaler)
            self.blocks_config["use_scaler"] = [True]

            param = np.zeros((scaler["param_size"]), np.uint8).view(np.float16)
            param[0] = 0.0 # header
            param[1] = 1.0 # alpha
            param[2] = 1.0 # bnw
            param[3] = 0.0 # bnb
            param[4] = 0.0 # header
            param[5] = 1.0 # alpha
            param[6] = 0.0 # header
            param[7] = 1.0 # alpha
            add_scaler_param(param.byteswap())

        else:
            self.blocks_config["use_scaler"] = [False]

    def _add_extra_info(self):
        self.blocks_config["input_format"] = []
        self.blocks_config["input_block"] = []
        self.blocks_config["input_shape"] = []
        self.blocks_config["output_block"] = []
        self.blocks_config["output_shape"] = []

        for block_config in self.blocks_config["blocks"]:
            if block_config["is_start"] == True:
                self.blocks_config["input_format"].append("image") # ex: image, temperature
                self.blocks_config["input_block"].append(block_config["index"])
                self.blocks_config["input_shape"].append(block_config["input_shape"])
            if block_config["is_logit"] == True:
                self.blocks_config["output_block"].append(block_config["index"])
                self.blocks_config["output_shape"].append(block_config["output_shape"] + [block_config["kernel_shape"][-1]])

        self.blocks_config["mode_gap"] = self.mode_gap
        self.blocks_config["multiple"] = self.multiple
        self.blocks_config["pad_order"] = self.default_pad_order
        self.blocks_config["pad_mode"] = self.default_pad_mode
        self.blocks_config["logit_order"] = self.logit_order
        self.blocks_config["use_scaler"] = self.use_scaler
        self.blocks_config["use_input_inv"] = self.use_input_inv
        self.blocks_config["use_opt_precision"] = self.use_opt_precision

    def _verify_block_config(self):
        for config in self.blocks_config["blocks"]:
            ''' parameter size check '''
            verify_parameter_size(config)

            ''' sram size check '''
            verify_sram_size(config)

    def _convert_block_to_instruction(self):
        '''
        op_type = BS3
         1. Calculation mode
         2. Shape(input, output, kernel)
         3. Padding(t,l,b,r)
         4. Stride(h,w)
         5. Options(Depth Separate, Bn, Relu, Xwn, etc)
        '''
        self.insts = {}

        for idx in range(len(self.blocks_config["blocks"])):
            inst = np.zeros((8), dtype=np.uint32)

            set_index(self.blocks_config, inst, idx)
            set_mode(self.blocks_config, inst, idx)
            set_xwn(self.blocks_config, inst, idx)
            set_input_shape(self.blocks_config, inst, idx)
            set_output_shape(self.blocks_config, inst, idx)
            set_kernel_shape(self.blocks_config, inst, idx)
            set_stride(self.blocks_config, inst, idx)
            set_padding(self.blocks_config, inst, idx)
            set_depth_separate_index(self.blocks_config, inst, idx)
            set_batch_noralization(self.blocks_config, inst, idx)
            set_residual(self.blocks_config, inst, idx)
            set_activate_function(self.blocks_config, inst, idx)
            set_scaler(self.blocks_config, inst, idx)

            self.insts[idx] = inst

    def _map_memory(self):
        map = {}
        map['out_offset'] = 0
        map['first_logit'] = 0
        map['logit_offset'] = 0
        for idx in range(self.blocks_config["n_block"]):
            map[idx] = {}
            set_input_addr(map, self.blocks_config, idx)
            set_param_addr(map, self.blocks_config, idx)
            set_resi_addr(map, self.blocks_config, idx)
            set_output_addr(map, self.blocks_config, idx)

            self.insts[idx][4] = map[idx]['input_addr']
            self.insts[idx][5] = map[idx]['param_addr']
            self.insts[idx][7] = map[idx]['output_addr']
            if 'residual_addr' in map[idx]:
                self.insts[idx][6] = map[idx]['residual_addr']

    def _merge_inst(self):
        instruction = np.zeros((8 * self.blocks_config["n_block"]), np.uint32).reshape(-1, 8)
        for idx in range(self.blocks_config["n_block"]):
            instruction[idx][:] = self.insts[idx][:]

        self.instruction = instruction.byteswap()
        self.parameter = np.fromfile('./param.bin', np.uint8)

    #################################
    ####### External - Methods
    #################################
    def create_blocks(self):
        # Layer
        self.refer = update_config(
            self.refer, self.input_shape,
            self.default_pad_order,
            self.default_pad_mode,
        )
        self.create_block_model()
        self.merge_block_model()
        # Block - Function
        self.translate_logit_order(self.logit_order)
        self.translate_mp_to_conv()
        self.translate_concat_to_conv()
        self.translate_gap_to_conv(self.mode_gap)
        self.translate_fc_to_conv()
        self.translate_group_conv()
        self.graph = update_config(
            self.graph, self.input_shape,
            self.default_pad_order,
            self.default_pad_mode,
        )
        self.convert_output_shapes()
        self.graph = update_config(
            self.graph, self.input_shape,
            self.default_pad_order,
            self.default_pad_mode,
        )
        # Block - Support device
        self.translate_align_data(multiple=self.multiple)
        self.graph = update_config(
            self.graph, self.input_shape,
            self.default_pad_order,
            self.default_pad_mode,
        )
        self.translate_split_conv(
            buffer_size=self.buffer_size, 
            multiple=self.multiple
        ) # from LAYERtoBLOCK
        self.graph = update_config(
            self.graph, self.input_shape,
            self.default_pad_order,
            self.default_pad_mode,
        )
        # Block - Optimize
        if self.use_input_inv: self.graph = invert_input_channel(self.graph)
        # self.report()
        return self

    def get_optimize_info(self, ref_graph):
        if self.use_opt_precision: self.graph = optimize_precision(self.graph)
        self.graph = optimize_param(self.graph, ref_graph, self.opt_cfg, use_dim=False)
        return self

    def create_parameters(self):
        self._update_binary_params()
        self._update_commands()
        self._execute_commands()
        return self

    def create_instruction(self):
        self._extract_block_config()
        self._append_scaler_block()
        self._add_constraint_option()
        self._add_extra_info()
        self._verify_block_config()
        self._convert_block_to_instruction()
        self._map_memory()
        self._merge_inst()

        return self

    def create_tachyrt_model(self):
        rt_model = tachyrt_model()
        self.runtime_model = rt_model.build_model(self.instruction, self.blocks_config, self.parameter)
        return self

        # def create_tachyrt_model(self):
        #     self._write_tachyrt()
        #     return self

    def report(self):
        for bi in self.graph.nodes:
            print("[INFO]: ------------- {} Block ------------".format(bi))
            print("[INFO]: name=", self.graph.nodes[bi]["name"])
            print("[INFO]: inputs=", self.graph.nodes[bi]["inputs"])
            print("[INFO]: outputs=", self.graph.nodes[bi]["outputs"])
            print("[INFO]: input_shapes=", self.graph.nodes[bi]["input_shapes"])
            print("[INFO]: output_shapes=", self.graph.nodes[bi]["output_shapes"])
            print("[INFO]: residual=", self.graph.nodes[bi]["residual"])
            print("[INFO]: activation=", self.graph.nodes[bi]["activation"])
            print("[INFO]: block_function=", self.graph.nodes[bi]["block_function"])
            print("[INFO]: channel_index=", self.graph.nodes[bi]["channel_index"])
            if "org_shapes" in self.graph.nodes[bi]:
                print("[INFO]: org_shapes=", self.graph.nodes[bi]["org_shapes"])

            if "pads_dynamic" in self.graph.nodes[bi]:
                print("[INFO]: pads_dynamic=", self.graph.nodes[bi]["pads_dynamic"])
            else:
                print("[INFO]: pads_dynamic=", "Empty")

            if "channel_residual" in self.graph.nodes[bi]:
                print("[INFO]: channel_residual=", self.graph.nodes[bi]["channel_residual"])
            else:
                print("[INFO]: channel_residual=", "Empty")

            print("[INFO]: number pipe =", len(self.graph.nodes[bi]["layers"]))
            for bti in self.graph.nodes[bi]["layers"]:
                print("[INFO]: {} NPU op=".format(bti), [layers["name"] for layers in self.graph.nodes[bi]["layers"][bti]])

            print("[INFO]: Parameter shapes=", [p.shape for p in self.graph.nodes[bi]["params"]])
            if "op_shapes" in self.graph.nodes[bi].keys():
                print("[INFO]: Operation Shapes=", self.graph.nodes[bi]["op_shapes"])

        print(self.graph)
        return self



if __name__ == "__main__":
    args = get_parser().parse_args()
    src_file = args.src_file
    dst_file = args.dst_file
    ref_file = args.ref_file
    shape = args.shape
    default_pad_order = args.default_pad_order
    default_pad_mode = args.default_pad_mode
    strategy_gap = args.strategy_gap
    logit_order = args.logit_order
    work_dir = args.work_dir
    script_dir = args.script_dir

    key = "tachy_model"
    tachy_model = tload(src_file)[key]
    tm = TachyBlock(
        tachy_model, shape, 
        default_pad_order,
        default_pad_mode,
        strategy_gap,
        logit_order,
        work_dir,
        script_dir,
    )
    tm = tm.create_blocks()

    if ref_file is not None:
        refer_model = tload(ref_file)[key]
        rm = TachyBlock(
            refer_model, shape, 
            default_pad_order,
            default_pad_mode,
            strategy_gap,
            logit_order,
            work_dir,
            script_dir,
        )
        rm = rm.create_blocks()
        tm = tm.get_optimize_info(rm.graph)
        tm = tm.create_parameters()
        tm = tm.create_instruction()
        tm = tm.create_tachyrt_model()
    
    tachy_dict = tdict()
    tachy_dict["tachy_model"] = tm
    tsave(dst_file, tachy_dict)
    # cfg = tm.get_param_config()
    # print(json.dumps(cfg, indent=4))

