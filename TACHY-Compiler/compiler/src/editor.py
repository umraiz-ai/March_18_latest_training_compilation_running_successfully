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
from collections import OrderedDict as od
import copy

import numpy as np
import networkx as nx

from graph import in_edges, sort_graph, update_edges, update_nodes, search_parents, search_children


_DEBUG = False


#################################
####### Internal - Layers
#################################
def _cascade_inout(inputs:list, outputs:list, n_in:int=2, prefix='dummy'):
    in_list = []
    layers = []
    for i, _input in enumerate(inputs):
        in_list.append(_input)
        if len(in_list) == n_in:
            if i != len(inputs) - 1: 
                layers.append([in_list, ['{}_{}'.format(prefix, i)]])
            else:
                layers.append([in_list, outputs])
            in_list = ['{}_{}'.format(prefix, i)]

    return layers

def _find_index_from_inout(name:str, graph:nx.DiGraph, key='inputs'):
    idx = None
    for i in graph.nodes:
        if name in graph.nodes[i][key]:
            idx = i

    return idx

def _hook_in_edges(edges:list, table:dict) -> list:
    result = []
    for i, o in edges:
        if i in table:
            result.append((table[i], o,))
        else:
            result.append((i, o,))
    return result

def _assign(graph_target, graph_ref):
    for i in graph_ref.nodes:
        graph_target.nodes[i].update(graph_ref.nodes[i])
    return graph_target

def _tuple2list(x):
    result = [] 
    for _x in x:
        result.append(list(_x))
    return result



#################################
####### External - Layers
#################################
def cascade_concat(dg:nx.DiGraph) -> nx.DiGraph:
    if len(dg.edges) == 0:
        return dg

    # Create New DiGraph
    dg_new = nx.DiGraph()
    # cnt = max([i for i in dg.nodes]) + 1
    cnt = max(dg.nodes) + 1
    edges = []
    hook_table = {}

    # Node
    for i in dg.nodes:
        name = dg.nodes[i]['name']
        op_type = dg.nodes[i]['op_type']
        l = len(dg.nodes[i]["inputs"])

        if op_type == 'Concat' and l > 2:
            layers = _cascade_inout(
                dg.nodes[i]['inputs'], 
                dg.nodes[i]['outputs'],
                prefix='{}_dummy'.format(name)
            )
            for n, (_in, _out) in enumerate(layers):
                dg_new.add_node(cnt)
                dg_new.nodes[cnt].update(dg.nodes[i])
                dg_new.nodes[cnt].update(
                    {'name': '{}_{}'.format(name, n), 'inputs':_in, 'outputs':_out}
                )

                o_edge = cnt
                for _in_name in _in:
                    i_edge = _find_index_from_inout(_in_name, dg, key='outputs')
                    if i_edge is None: i_edge = cnt - 1
                    edges.append((i_edge, o_edge))

                cnt += 1

            hook_table[i] = cnt - 1
        else: 
            dg_new.add_node(i)
            dg_new.nodes[i].update(dg.nodes[i])
            _in_edges = in_edges(dg.edges, i)
            edges += _hook_in_edges(_in_edges, hook_table)

    dg_new.add_edges_from(edges)
    graph = _assign(sort_graph(dg_new, edges), dg_new)

    return graph





#################################
####### Internal - Blocks
#################################
def _use_scale(node:dict) -> bool:
    w = node["params"][1]
    b = node["params"][2]
    use_w = np.all(w == 1)
    use_b = np.all(b == 0)
    if use_w and use_b:
        return False
    else:
        return True
            
def _extra_block(
    node:dict, 
    cnt:int, 
    inputs:list, 
    input_shapes:list,
    channel_index:list=None,
    dtype:str="float32"
) -> dict:
    extra_inputs = node["inputs"][1:2] if len(node["inputs"]) > 1 else []
    extra_input_shapes = node["input_shapes"][1:2] if len(node["input_shapes"]) > 1 else []

    _node = create_node(cnt)

    # inout
    _node["inputs"] = inputs + extra_inputs
    _node["input_shapes"] = input_shapes + extra_input_shapes
    _node["outputs"] = node["outputs"] 
    _node["output_shapes"] = node["output_shapes"]

    # params
    _node["params"] = [
        identical_kernel(node["params"][0].shape[-1], dtype=dtype),
        node["params"][1],
        node["params"][2],
    ]

    # basic
    _node["residual"] = node["residual"]
    _node["activation"] = node["activation"]
    _node["configs"].update({
        "kernel_shape": (1, 1),
        "strides": (1, 1),
        "pads": (0, 0, 0, 0),
    })
    _node["layers"].update(node["layers"])

    # additional
    if channel_index is None:
        _node["channel_index"] = node["channel_index"]
    else:
        _node["channel_index"] = channel_index
    _node["channel_residual"] = node["channel_residual"]
    _node["block_function"] = "IdenticalConv"
    return _node

def _get_dummy_pads(node:dict, limit:int=8):
    # Input information
    _, _, iw, _ = node["input_shapes"][0]
    ic = node["channel_index"][1] - node["channel_index"][0]
    # Output information
    _, _, ow, _ = node["output_shapes"][0]

    stride_w = node["configs"]["strides"][1]
    pads_i = max((limit - ow) * stride_w - iw, 0)
    pads_o = max(limit - ow, 0) 
    return pads_i, pads_o

def _require_input_width(node:dict, limit:int=8, margin:int=1):
    _, _, iw, _ = node["input_shapes"][0]
    is_transpose = True if node["block_function"] == "ConvTranspose" else False

    pad_iw, _ = _get_dummy_pads(node, limit=limit)
    dynamic_pad_w = node["pads_dynamic"][1][0] + node["pads_dynamic"][1][1]
    pad_w = max(dynamic_pad_w, pad_iw)

    w = max(limit, iw + pad_w if not is_transpose else 2 * iw + pad_w) + margin
    return w

def _require_buffer_size(node:dict, limit:int=8):
    # Input information
    ic = node["channel_index"][1] - node["channel_index"][0]
    buffer_size = _require_input_width(node, limit) * ic

    return buffer_size

def _over_buffer(node:dict, buffer_size:int, limit:int=8) -> bool:
    return True if _require_buffer_size(node, limit) > buffer_size else False 
    
def _get_step_list(step, n_ch, base=0) -> list:
    result = []
    n_iter = n_ch // step
    n_extra = True if n_ch % step > 0 else False
    result = [(i*step+base, (i+1)*step+base, n_ch) for i in range(n_iter)]
    if n_extra:
        n_base = n_iter * step
        result.append((n_base, n_ch, n_ch))

    return result
        
def _get_step_list_with_channel(step, n_ch, i_ch, base=0) -> list:
    result = []
    n_iter = n_ch // step
    n_extra = True if n_ch % step > 0 else False
    result = [(i*step+base, (i+1)*step+base, i_ch) for i in range(n_iter)]
    if n_extra:
        n_base = n_iter * step + base
        result.append((n_base, n_ch + base, i_ch))

    return result
        


#################################
####### Internal - Concat -> Conv Core
#################################
def _concat_kernel(ch_0:int, ch_1:int, first:bool, dtype:str="float32"):
    ksize = (1, 1, ch_0, ch_0 + ch_1)
    kernel = np.zeros(ksize, dtype=dtype)
    ik = identical_kernel(ch_0) # (1, 1, ch_0, ch_0)
    if first:
        kernel[..., :ch_0] = ik
    else:
        kernel[..., ch_1:] = ik
    return kernel

def _concat_to_conv(node, cnt, dtype="float32"):
    assert len(node["input_shapes"]) == 2
    shape_0, shape_1 = node["input_shapes"]
    ch_0, ch_1 = shape_0[-1], shape_1[-1]

    kernel_0 = _concat_kernel(ch_0, ch_1, True, dtype)
    kernel_1 = _concat_kernel(ch_1, ch_0, False, dtype)

    dummy_edge_name = "{}_{}".format(node["name"], cnt)
    node_0 = create_node(cnt)
    node_0["inputs"] = node["inputs"][0:1]
    node_0["input_shapes"] = node["input_shapes"][0:1]
    node_0["outputs"] = [dummy_edge_name] # Dummy
    node_0["output_shapes"] = node["output_shapes"]
    node_0["params"] = [kernel_0, identical_weight(ch_0 + ch_1), identical_bias(ch_0 + ch_1)]
    node_0["configs"].update({
        "kernel_shape": (1, 1),
        "strides": (1, 1),
        "pads": (0, 0, 0, 0),
    })
    node_0["residual"] = False
    node_0["layers"].update(node["layers"])
    node_0["block_function"] = "IdenticalConv"
    node_0["channel_index"] = (
        0, 
        node["input_shapes"][0][-1],
        node["input_shapes"][0][-1],
    )

    cnt += 1
    node_1 = create_node(cnt)
    node_1["inputs"] = node["inputs"][1:2] + [dummy_edge_name]
    node_1["input_shapes"] = node["input_shapes"][1:2] + node["output_shapes"]
    node_1["outputs"] = node["outputs"]
    node_1["output_shapes"] = node["output_shapes"]
    node_1["params"] = [kernel_1, identical_weight(ch_0 + ch_1), identical_bias(ch_0 + ch_1)]
    node_1["configs"].update({
        "kernel_shape": (1, 1),
        "strides": (1, 1),
        "pads": (0, 0, 0, 0),
    })
    node_1["residual"] = True
    node_1["layers"].update(node["layers"])
    node_1["block_function"] = "IdenticalConv"
    node_1["channel_index"] = (
        0, 
        node["input_shapes"][1][-1],
        node["input_shapes"][1][-1],
    )

    return [node_0, node_1]

#################################
####### Internal - Chain GAP Core
#################################
def _gap_param(channel:int, k:list, denominator:int, dtype:str="float32") -> list:
    kernel = identical_kernel(channel, k=k, dtype=dtype) # (kh, kw, c, c)
    weight = identical_weight(channel, dtype=dtype) / denominator
    bias = identical_bias(channel, dtype=dtype)

    return [kernel, weight, bias]

def _gap_to_conv(
    node, 
    cnt, 
    dtype="float32", 
):
    blocks = []
    b, h, w, c = node["input_shapes"][0]
    gap_cnt = 0
    last = False
    inputs = node["inputs"]
    input_shapes = node["input_shapes"]

    locked_dim = node["configs"]["locked_dim"] if node["configs"]["op_dim"] == 1 else None
    if locked_dim is None:
        k = (2, 2)
        limits = (1, 1)
    elif locked_dim == "w":
        k = (2, 1)
        limits = (1, w)
    elif locked_dim == "h":
        k = (1, 2)
        limits = (h, 1)

    kh, kw = k
    limit_h, limit_w = limits

    while True:
        done_h = True if h == limit_h else False
        done_w = True if w == limit_w else False

        pad_h = kh - (h % kh) if h % kh > 0 else 0
        pad_w = kw - (w % kw) if w % kw > 0 else 0
        pad_h = 0 if done_h else pad_h
        pad_w = 0 if done_w else pad_w

        kh = 1 if done_h else kh
        kw = 1 if done_w else kw

        oh = (h + pad_h) // kh
        ow = (w + pad_w) // kw
        last = True if oh == limit_h and ow == limit_w else False

        if last:
            outputs = node["outputs"]
            # output_shapes = [(b, oh, ow, c)]
            output_shapes = [(b, limit_h, limit_w, c)]
        else:
            outputs = ["{}_{}".format(node["name"], gap_cnt)]
            output_shapes = [(b, oh, ow, c)]

        # Create node
        _node = create_node(cnt + gap_cnt)
        if last:
            if len(node["inputs"]) == 2:
                extra_input = node["inputs"][1:2]
                extra_input_shape = node["input_shapes"][1:2]
            else: 
                extra_input = []
                extra_input_shape = []

            _node["inputs"] = inputs + extra_input
            _node["input_shapes"] = input_shapes + extra_input_shape

            ksize = (node["input_shapes"][0][1] / limit_h) * (node["input_shapes"][0][2] / limit_w)
            _node["params"] = _gap_param(
                node["input_shapes"][0][-1], 
                (kh, kw), 
                ksize, 
                dtype
            )
            _node["residual"] = True if len(extra_input) > 0 else False

        else:
            _node["inputs"] = inputs
            _node["input_shapes"] = input_shapes
            _node["params"] = _gap_param(
                input_shapes[0][-1], 
                (kh, kw), 
                1, 
                dtype
            )
            _node["residual"] = False

        _node["outputs"] = outputs 
        _node["output_shapes"] = output_shapes

        _node["channel_index"] = node["channel_index"]

        _node["configs"].update({
            "kernel_shape": (kh, kw),
            "strides": (kh, kw),
            "pads": (0, 0, pad_h, pad_w),
            "auto_pad": "SAME_UPPER".encode("utf-8"),
        })
        _node["layers"].update(node["layers"])
        _node["block_function"] = "IdenticalConv"
        
        blocks.append(_node)

        inputs = outputs
        input_shapes = output_shapes
        if last: break

        b, h, w, c = input_shapes[0]
        gap_cnt += 1

    return blocks

#################################
####### Internal - Serial Conv Core
#################################
def _serial_conv_param(
    indices:np.ndarray, 
    in_channel:int,
    out_channel:int,
    ksize:float,
    dtype:str="float32"
) -> list:
    kernel = np.zeros((in_channel, out_channel), dtype=dtype)
    for i, ind in enumerate(indices): 
        kernel[i, ind] = 1
    weight = np.ones((out_channel,), dtype=dtype) / ksize
    bias = identical_bias(out_channel, dtype=dtype)
    return [kernel[None, None, ...], weight, bias]

def _gap_to_serial_conv(node, cnt, buffer_block_size=512, dtype="float32"):
    blocks = []
    # b, h, w, c = node["input_shapes"][0]
    b, h, w, _ = node["input_shapes"][0]
    ch_base = node["channel_index"][0]
    c = node["channel_index"][1] - node["channel_index"][0]

    ksize = float(h * w)
    ch_ind_matrix = (np.ones((h, w, c), dtype="int32") * np.arange(c)).transpose(0,2,1).reshape(-1) # (H,C,W)
    n = ch_ind_matrix.size
    step = buffer_block_size * min(1, w)
    extra = use_block_function(1, node) or use_block_function(2, node) or use_block_function(3, node)

    gap_cnt = 0
    last = False
    first = True

    while True:
        last = True if step * (gap_cnt + 1) + ch_base >= n else False
        # Create node
        _node = create_node(cnt + gap_cnt)
        outputs = ["{}_{}".format(node["name"], gap_cnt)]
        output_shapes = node["output_shapes"]

        if last:
            _node["inputs"] = node["inputs"][0:1] + inputs
            _node["input_shapes"] = [(b, 1, 1, n - (gap_cnt * step))]
            _node["residual"] = True
            _node["channel_index"] = [gap_cnt * step + ch_base, n, n]
            _node["params"] = _serial_conv_param(
                ch_ind_matrix[gap_cnt * step + ch_base:n + ch_base],
                n - (gap_cnt * step) + ch_base,
                c, 
                ksize,
                dtype
            )

        elif first:
            first = False
            _node["inputs"] = node["inputs"][0:1]
            _node["input_shapes"] = [(b, 1, 1, step)]
            _node["residual"] = False
            _node["channel_index"] = [
                gap_cnt * step + ch_base, 
                (gap_cnt + 1) * step + ch_base, 
                n
            ]
            _node["params"] = _serial_conv_param(
                ch_ind_matrix[gap_cnt * step + ch_base:(gap_cnt + 1) * step + ch_base],
                step, c,
                ksize,
                dtype
            )
        else:
            _node["inputs"] = node["inputs"][0:1] + inputs
            _node["input_shapes"] = [(b, 1, 1, step)]
            _node["residual"] = True
            _node["channel_index"] = [
                gap_cnt * step + ch_base, 
                (gap_cnt + 1) * step + ch_base, 
                n
            ]
            _node["params"] = _serial_conv_param(
                ch_ind_matrix[gap_cnt * step + ch_base:(gap_cnt + 1) * step + ch_base],
                step, c,
                ksize,
                dtype
            )

        _node["outputs"] = outputs 
        _node["output_shapes"] = output_shapes
        _node["org_shapes"] = node["org_shapes"] if "org_shapes" in node else node["output_shapes"]

        _node["block_function"] = "SerialConv"
        _node["configs"].update({
            "kernel_shape": (1, 1),
            "strides": (1, 1),
            "pads": (0, 0, 0, 0),
        })
        _node["layers"].update(node["layers"])
            
        # Extra
        if last: 
            if extra:
                blocks.append(_node)
                # Extra node
                blocks.append(_extra_block(node, cnt + gap_cnt + 1, outputs, output_shapes))
            else:
                # Hook
                _node["outputs"] = node["outputs"] 
                _node["output_shapes"] = node["output_shapes"]
                blocks.append(_node)
            break

        else:
            blocks.append(_node)

        # Update
        inputs = outputs
        input_shapes = output_shapes
        gap_cnt += 1

    return blocks

#################################
####### Internal - Split Conv Core
#################################
def _split_conv_param(
    params:list, 
    indices:tuple, 
    dtype:str="float32"
) -> list:
    stt, end = indices
    k, w, b = params
    _, _, _, channel = k.shape
    kernel = k[:, :, stt:end, :] # (kh, kw, step, o)
    # weight = np.ones_like(w, dtype=dtype) 
    # bias = np.zeros_like(b, dtype=dtype) 
    return [kernel, None, None]

def _get_split_conv_node(
    node:dict, 
    cnt:int, i:int,
    inputs:list,
    input_shapes:list,
    outputs:list,
    output_shapes:list,
    channel_index:list,
    dtype:str="float32",
) -> dict:

    # Fixed information

    # Create node
    _node = create_node(cnt + i)

    # inout
    _node["inputs"] = inputs
    _node["input_shapes"] = input_shapes
    _node["outputs"] = outputs 
    _node["output_shapes"] = output_shapes
        
    indices = (0, channel_index[1] - channel_index[0]) if channel_index[1] > node["params"][0].shape[2] else channel_index[:2]
    # params
    _node["params"] = _split_conv_param(
        node["params"],
        indices,
        dtype
    )
    _node["params"][1] = node["params"][1]
    _node["params"][2] = node["params"][2]

    # basic
    _node["residual"] = True if len(inputs) > 1 else False
    _node["activation"] = node["activation"]
    _node["configs"].update(node["configs"])
    _node["layers"].update(node["layers"])

    # additional
    _node["channel_index"] = channel_index
    _node["channel_residual"] = node["channel_residual"]

    # is_transpose = True if node["block_function"] == "ConvTranspose" else False
    # _node["block_function"] = "ConvTranspose" if is_transpose else "Conv"
    _node["block_function"] = node["block_function"]
    _node["org_shapes"] = node["org_shapes"] if "org_shapes" in node else node["output_shapes"]

    return _node

def _split_conv(
    node:dict, 
    cnt:int, 
    buffer_size:int=4096, 
    limit:int=8, 
    multiple:int=8, 
    dtype:str="float32"
) -> list:

    blocks = []
    if len(node["inputs"]) == 2:
        residual_inputs = node["inputs"][1:]
        residual_input_shapes = node["input_shapes"][1:]
    else:
        residual_inputs, residual_input_shapes = [], []
    c = node["input_shapes"][0][-1]

    step = int(buffer_size / _require_input_width(node, limit))
    assert step > 0
    if step > multiple:
        step -= step % multiple
        # step += multiple - (step % multiple)
    else:
        step = multiple
    indices = _get_step_list_with_channel(step, node["channel_index"][1] - node["channel_index"][0], c, base=node["channel_index"][0])

    # Nodes
    for i, ch_index in enumerate(indices):
        # Split index information
        i_channel = ch_index[1] - ch_index[0]
        if "org_shapes" in node:
            o_channel = node["org_shapes"][0][-1]
        else:
            o_channel = node["output_shapes"][0][-1]

        # Input
        inputs = node["inputs"][0:1] + residual_inputs
        input_shapes = node["input_shapes"][0:1] + residual_input_shapes
        input_shapes = _tuple2list(input_shapes)
        input_shapes[0][-1] = i_channel

        # Output
        if i == len(indices) - 1: # End
            outputs = node["outputs"]
            output_shapes = node["output_shapes"]
        else:
            outputs = ["{}_sc_{}".format(node["name"], i)]
            if "org_shapes" in node:
                output_shapes = node["org_shapes"][0:1]
            else:
                output_shapes = node["output_shapes"][0:1]

        # Params & Activation
        node_hooked = copy.deepcopy(node)
        if i < len(indices) - 1: # non end
            node_hooked["params"][2] = identical_bias(o_channel, dtype=dtype)
            node_hooked["activation"] = None

        _node = _get_split_conv_node(
            node_hooked,
            cnt, i,
            inputs,
            input_shapes,
            outputs,
            output_shapes,
            ch_index,
            dtype,
        )
        blocks.append(_node)

        residual_inputs = outputs
        residual_input_shapes = output_shapes

    # ##### Dont use extra block in split conv #####
    # # Extra node
    # extra = use_block_function(1, node) or use_block_function(2, node) or use_block_function(3, node)
    # if extra:
    #     oc = node["output_shapes"][0][-1]
    #     blocks.append(
    #         _extra_block(
    #             node, 
    #             cnt + len(indices), 
    #             residual_inputs, 
    #             residual_input_shapes,
    #             channel_index=(0, oc, oc),
    #             dtype=dtype
    #         )
    #     )
    # else:
    #     blocks[-1]["outputs"] = node["outputs"]
    #     blocks[-1]["output_shapes"] = node["output_shapes"]

    return blocks


#################################
####### Internal - FC -> Conv Core
#################################
def _fc_to_conv(
    node, 
    cnt, 
    dtype="float32", 
):
    # kernel extension
    node["params"][0] = node["params"][0][None, None, ...]
    node["input_shapes"][0] = [1, 1] + list(node["input_shapes"][0]) # TODO: to height
    node["output_shapes"][0] = [1, 1] + list(node["output_shapes"][0]) # TODO: to height
    node["block_function"] = "Conv"
    node["configs"].update({
        "kernel_shape": (1, 1),
        "strides": (1, 1),
        "pads": (0, 0, 0, 0),
    })
    node["reshape"] = node["input_shapes"][0]

    return node

#################################
####### Internal - Output Reshape
#################################
def _get_reshape(
    idx:int,
    graph:nx.DiGraph, 
) -> list:
    children = search_children(graph, idx)
    for ni in children:
        if "reshape" in graph.nodes[ni]:
            return graph.nodes[ni]["reshape"]
    return None
    
def _reshape_outputs(
    node:dict, 
    shape:list,
    cnt:int, 
    dtype="float32", 
):
    node["op_shapes"] = node["output_shapes"].copy()
    node["output_shapes"][0] = shape
    return node


################
####### Internal - Grouped Conv Core
################
def _is_group_conv(node:dict) -> bool:
    # return True if node["configs"]["groups"] > 1 else False
    return True if node["configs"]["group"] > 1 else False

def _group_conv_param(
    params:list,
    input_indices:tuple,
    output_indices:tuple,
    keep:bool=True,
    dtype:str="float32"
) -> list:
    '''
    args: keep / whether to maintain the shape of parameter or not
    '''
    i_stt, i_end = input_indices
    o_stt, o_end = output_indices
    kernel, weight, bias = params
    
    if keep:
        kernel_masked = np.zeros_like(kernel)
        # kernel_masked[:, :, i_stt:i_end, o_stt:o_end] = kernel[:, :, i_stt:i_end, o_stt:o_end]
        kernel_masked[:, :, :, o_stt:o_end] = kernel[:, :, :, o_stt:o_end]

        weight_masked = np.ones_like(weight)
        # weight_masked[o_stt:o_end] = weight[o_stt:o_end]

        bias_masked = np.zeros_like(bias)
        # bias_masked[o_stt:o_end] = bias[o_stt:o_end]
    else:
        # kernel_masked = kernel[:, :, i_stt:i_end, o_stt:o_end]
        kernel_masked = kernel[:, :, :, o_stt:o_end]
        weight_masked = np.ones_like(weight[o_stt:o_end])
        bias_masked = np.zeros_like(bias[o_stt:o_end])

    return [kernel_masked, weight_masked, bias_masked]

def _get_group_conv_node(
    node:dict, 
    cnt:int, i:int,
    inputs:list,
    input_shapes:list,
    outputs:list,
    output_shapes:list,
    i_ch_index:list,
    o_ch_index:list,
    configs:dict=None,
    keep:bool=True,
    dtype:str="float32",
) -> dict:

    # Fixed information

    # Create node
    _node = create_node(cnt + i)

    # inout
    _node["inputs"] = inputs
    _node["input_shapes"] = input_shapes
    _node["outputs"] = outputs 
    _node["output_shapes"] = output_shapes

    # params
    _node["params"] = _group_conv_param(
        node["params"],
        i_ch_index[:2],
        o_ch_index[:2],
        keep=keep,
        dtype=dtype
    )

    # basic
    _node["residual"] = True if len(inputs) > 1 else False 
    _node["activation"] = None
    _node["configs"].update(node["configs"])
    if configs is not None:
        _node["configs"].update(configs)
    _node["layers"].update(node["layers"])

    # additional
    _node["channel_index"] = i_ch_index
    _node["channel_residual"] = node["channel_residual"]

    # is_transpose = True if node["block_function"] == "ConvTranspose" else False
    # _node["block_function"] = "ConvTranspose" if is_transpose else "Conv"
    _node["block_function"] = node["block_function"]

    return _node


def _group_conv(
    node:dict, 
    cnt:int,
    dtype:str="float32",
    keep:bool=True,
) -> list:
    blocks = []
    residual_inputs, residual_input_shapes = [], []

    # n_split = node["configs"]["groups"]
    n_split = node["configs"]["group"]
    assert n_split > 1

    _, _, _, ic = node["input_shapes"][0]
    _, _, _, oc = node["output_shapes"][0]
    i_step = ic // n_split
    o_step = oc // n_split
    assert i_step * n_split == ic
    assert o_step * n_split == oc

    i_indices = _get_step_list(i_step, ic)
    o_indices = _get_step_list(o_step, oc)

    for i, (i_ch_index, o_ch_index) in enumerate(zip(i_indices, o_indices)):
        # Split index information
        i_channel = i_ch_index[1] - i_ch_index[0]
        inputs = node["inputs"][0:1] + residual_inputs
        input_shapes = node["input_shapes"][0:1] + residual_input_shapes
        input_shapes = _tuple2list(input_shapes)
        # input_shapes[0][-1] = i_channel

        o_channel = o_ch_index[1] - o_ch_index[0]
        outputs = ["{}_gc_{}".format(node["name"], i)]
        output_shapes = node["output_shapes"][0:1]
        output_shapes = _tuple2list(output_shapes)
        output_shapes[0][-1] = oc if keep else o_channel

        _node = _get_group_conv_node(
            node,
            cnt, i,
            inputs,
            input_shapes,
            outputs,
            output_shapes,
            i_ch_index,
            o_ch_index,
            configs=None,
            keep=keep,
            dtype=dtype,
        )
        blocks.append(_node)

        residual_inputs = outputs
        residual_input_shapes = output_shapes

    # Extra node
    extra = use_block_function(1, node) or use_block_function(2, node) or use_block_function(3, node)
    if extra:
        oc = node["output_shapes"][0][-1]
        blocks.append(
            _extra_block(
                node, 
                cnt + len(i_indices), 
                residual_inputs if len(node["inputs"]) == 1 else residual_inputs + node["inputs"][1:2] , 
                node["input_shapes"],
                channel_index=(0, oc, oc),
                dtype=dtype
            )
        )
    else:
        blocks[-1]["outputs"] = node["outputs"]
        blocks[-1]["output_shapes"] = node["output_shapes"]

    return blocks

#################################
####### Internal - Split Maxpool Core
#################################
def _split_maxpool_param(
    params:list, 
    indices:tuple, 
    dtype:str="float32"
) -> list:
    stt, end = indices
    kernel, _, _ = params
    _, _, _, channel = kernel.shape
    kernel = kernel[:, :, stt:end, :] # (kh, kw, step, o)
    return [kernel, None, None]

def _split_mp_param(
    params:list,
    input_indices:tuple,
    output_indices:tuple,
    keep:bool=False,
    dtype:str="float32"
) -> list:
    '''
    args: keep / whether to maintain the shape of parameter or not
    '''
    i_stt, i_end = input_indices
    o_stt, o_end = output_indices
    kernel, weight, bias = params
    
    if keep:
        kernel_masked = np.zeros_like(kernel)
        kernel_masked[:, :, i_stt:i_end, o_stt:o_end] = kernel[:, :, i_stt:i_end, o_stt:o_end]

        weight_masked = np.zeros_like(weight)
        weight_masked[o_stt:o_end] = weight[o_stt:o_end]

        bias_masked = np.zeros_like(bias)
        bias_masked[o_stt:o_end] = bias[o_stt:o_end]
    else:
        kernel_masked = kernel[:, :, i_stt:i_end, o_stt:o_end]
        weight_masked = weight[o_stt:o_end]
        bias_masked = bias[o_stt:o_end]

    return [kernel_masked, weight_masked, bias_masked]

def _get_split_mp_node(
    node:dict, 
    cnt:int, i:int,
    inputs:list,
    input_shapes:list,
    outputs:list,
    output_shapes:list,
    i_ch_index:list,
    o_ch_index:list,
    configs:dict=None,
    keep:bool=False,
    dtype:str="float32",
) -> dict:

    # Fixed information

    # Create node
    _node = create_node(cnt + i)

    # inout
    _node["inputs"] = inputs
    _node["input_shapes"] = input_shapes
    _node["outputs"] = outputs 
    _node["output_shapes"] = output_shapes

    # params
    _node["params"] = _split_mp_param(
        node["params"],
        i_ch_index[:2],
        o_ch_index[:2],
        keep=keep,
        dtype=dtype
    )

    # basic
    _node["residual"] = True if len(inputs) > 1 else False 
    _node["activation"] = None
    _node["configs"].update(node["configs"])
    if configs is not None:
        _node["configs"].update(configs)
    _node["layers"].update(node["layers"])

    # additional
    _node["channel_index"] = i_ch_index
    _node["channel_residual"] = node["channel_residual"]

    # is_transpose = True if node["block_function"] == "ConvTranspose" else False
    # _node["block_function"] = "ConvTranspose" if is_transpose else "Conv"
    _node["block_function"] = node["block_function"]

    return _node


def _split_maxpool(
    node:dict, 
    cnt:int, 
    buffer_size:int=4096, 
    limit:int=8, 
    multiple:int=8, 
    dtype:str="float32"
) -> list:

    blocks = []
    pre_residual_inputs, pre_residual_input_shapes = [], []
    residual_inputs, residual_input_shapes = [], []
    concat_cnt = 0
    _, _, _, c = node["input_shapes"][0]

    step = int(buffer_size / _require_input_width(node, limit))
    assert step > 0
    if step > multiple:
        step -= step % multiple
        # step += multiple - (step % multiple)
    else:
        step = multiple
    indices = _get_step_list(step, c, base=node["channel_index"][0])

    # Nodes
    for i, ch_index in enumerate(indices):
        # Conv
        # Split index information
        n_channel = ch_index[1] - ch_index[0]
        conv_inputs = node["inputs"][0:1]
        conv_input_shapes = node["input_shapes"][0:1]
        conv_input_shapes = _tuple2list(conv_input_shapes)
        conv_input_shapes[0][-1] = n_channel

        conv_outputs = ["{}_mp0_{}".format(node["name"], i)]
        conv_output_shapes = node["input_shapes"][0:1]
        conv_output_shapes = _tuple2list(conv_output_shapes)
        conv_output_shapes[0][-1] = n_channel

        conv_i_ch_index = ch_index
        conv_o_ch_index = ch_index

        conv_configs = {
            "kernel_shape": (1, 1),
            "strides": (1, 1),
            "pads": (0, 0, 0, 0),
        }
        # Node
        _node = _get_split_mp_node(
            node,
            cnt + i + concat_cnt, i,
            conv_inputs,
            conv_input_shapes,
            conv_outputs,
            conv_output_shapes,
            conv_i_ch_index,
            conv_o_ch_index,
            configs=conv_configs,
            dtype=dtype,
        )

        _node["block_function"] = "IdenticalConv"
        blocks.append(_node)

        # MaxPool
        # Split index information
        # mp_inputs = conv_outputs + residual_inputs
        # mp_input_shapes = conv_output_shapes + residual_input_shapes
        mp_inputs = conv_outputs
        mp_input_shapes = conv_output_shapes

        mp_outputs = ["{}_mp1_{}".format(node["name"], i)]
        mp_output_shapes = node["output_shapes"][0:1]
        mp_output_shapes = _tuple2list(mp_output_shapes)
        mp_output_shapes[0][-1] = n_channel

        # Node
        _node = init_node(
            node,
            cnt + i + concat_cnt + 1, i,
            mp_inputs,
            mp_input_shapes,
            mp_outputs,
            mp_output_shapes,
            'MaxPool',
            dtype=dtype,
        )
        blocks.append(_node)

        # For next
        if i == 1:
            pre_residual_inputs = residual_inputs
            pre_residual_input_shapes = residual_input_shapes
        elif i > 1:
            pre_residual_inputs = cc_outputs
            pre_residual_input_shapes = cc_output_shapes

        residual_inputs = mp_outputs
        residual_input_shapes = mp_output_shapes

        # Concat
        if i > 0:
            # Split index information
            cc_inputs = pre_residual_inputs + residual_inputs
            cc_input_shapes = pre_residual_input_shapes + residual_input_shapes

            if i == len(indices) - 1:
                cc_outputs = node["outputs"]
                cc_output_shapes = node["output_shapes"]
            else:
                cc_outputs = ["{}_mp2_{}".format(node["name"], i)]
                cc_output_shapes = copy.deepcopy(mp_output_shapes)
                cc_output_shapes[0][-1] = pre_residual_input_shapes[0][-1] + residual_input_shapes[0][-1]
            # Node
            _node = init_node(
                node,
                cnt + i + concat_cnt + 2, i,
                cc_inputs,
                cc_input_shapes,
                cc_outputs,
                cc_output_shapes,
                'Concat',
                dtype=dtype,
            )
            blocks.append(_node)
            concat_cnt += 1

    # Extra node
    extra = use_block_function(1, node) or use_block_function(2, node) or use_block_function(3, node)
    if extra:
        blocks.append(
            _extra_block(
                node, 
                cnt + len(indices), 
                residual_inputs, 
                residual_input_shapes
            )
        )
    else:
        blocks[-1]["outputs"] = node["outputs"]
        blocks[-1]["output_shapes"] = node["output_shapes"]

    return blocks


#################################
####### Internal - Merge Block Core
#################################
def _compare_layers(
    node_0:dict,
    node_1:dict,
) -> bool:
    layers_0 = node_0["layers"]
    layers_1 = node_1["layers"]
    phase_0 = True
    phase_1 = False
    for i in range(4):
        l0 = layers_0[i]
        l1 = layers_1[i]

        if i == 0:
            if len(l0) == 0:
                return False
        else:
            if len(l1) > 0:
                phase_0 = False
                phase_1 = True

        if phase_0:
            if len(l1) > 0:
                return False

        if phase_1:
            if len(l0) > 0:
                return False

    return True


def _check_merge_block(
    idx:int,
    graph:nx.DiGraph, 
) -> list:
    children = search_children(graph, idx)
    for ni in children:
        if _compare_layers(graph.nodes[idx], graph.nodes[ni]):
            return graph.nodes[ni], ni

    return None, None

def _merge_block_node(
    node_0:dict, 
    node_1:dict, 
    cnt:int,
) -> dict:
    # inout
    if use_block_function(2, node_1):
        inputs = node_1["inputs"]
        input_shapes = node_1["input_shapes"]
        inputs[0] = node_0["inputs"][0]
        input_shapes[0] = node_0["input_shapes"][0]
        node_0["inputs"] = inputs
        node_0["input_shapes"] = input_shapes

    outputs = node_1["outputs"]
    output_shapes = node_1["output_shapes"]
    node_0["outputs"] = outputs 
    node_0["output_shapes"] = output_shapes

    # params
    if use_block_function(1, node_1):
        node_0["params"][1] = node_1["params"][1]
        node_0["params"][2] = node_1["params"][2]

    # basic
    node_0["residual"] = True if len(node_0["inputs"]) > 1 else False
    if use_block_function(3, node_1):
        node_0["activation"] = node_1["activation"]

    node_1["configs"].update(node_0["configs"])
    node_0["configs"] = node_1["configs"] 
    
    layer_dict = {}
    for i in range(len(node_0["layers"])):
        l0 = node_0["layers"][i]
        l1 = node_1["layers"][i]
        layer_dict[i] = l0 + l1
    node_0["layers"] = layer_dict

    # for bti in node_0["layers"]:
    #     print("[INFO]: {} Block-End =".format(bti), [layers["name"] for layers in node_0["layers"][bti]])
    return node_0



#################################
####### External - Blocks
#################################
def create_node(cnt, device="BS3"):
    node = {
        "op_type": device,
        "name": "Block_{}".format(cnt),
        "layers": {
            0: [], # Single
            1: [], # Multi
            2: [], # Single
            3: [], # Single
            4: [], # Single
        },
        "inputs": [],
        "outputs": [],
        "input_shapes": [],
        "output_shapes": [],
        "configs": {
            "kernel_shape": (1, 1),
            "strides": (1, 1),
            "pads": (0, 0, 0, 0),
            "op_dim": 2,
            # "groups": 1,
            "group": 1,
        },
        "residual": False, # True, False
        "activation": None, # Alpha, None
        "channel_index": None, # List, None
        "channel_residual": (0, 0), # (in_channel, out_channel)
        "params": [], # Kernel, Weight, Bias
        "block_function": None, # None, List:["Conv", "ConvTranspose", "Gemm", "MaxPool", "GlobalAveragePool", "Concat"]
    }
    return node

def init_node(
    node:dict, 
    cnt:int, i:int,
    inputs:list,
    input_shapes:list,
    outputs:list,
    output_shapes:list,
    function:str,
    channel_index:list=None,
    dtype:str="float32",
) -> dict:

    # Fixed information
    # assert input_shapes[0][-1] == output_shapes[0][-1]
    if channel_index is None:
        channel_index = (0, input_shapes[0][-1], input_shapes[0][-1])
    # print(node["name"], input_shapes[0][-1], output_shapes[0][-1])

    # Create node
    _node = create_node(cnt + i)

    # inout
    _node["inputs"] = inputs
    _node["input_shapes"] = input_shapes
    _node["outputs"] = outputs 
    _node["output_shapes"] = output_shapes
    _node["org_shapes"] = output_shapes
        
        
    # params
    c = channel_index[1] - channel_index[0]
    _node["params"] = [
        identical_kernel(c),
        identical_weight(c),
        identical_bias(c),
    ]

    # basic
    _node["residual"] = True if len(inputs) > 1 else False
    _node["activation"] = None
    _node["configs"].update(node["configs"])
    _node["layers"].update(node["layers"])

    # additional
    _node["channel_index"] = channel_index
    _node["channel_residual"] = node["channel_residual"]
    _node["block_function"] = function

    return _node

def identical_kernel(n_channel:int, k:tuple=(1,1), dtype:str="float32") -> np.ndarray:
    kh, kw = k
    kernel = np.zeros((kh, kw, n_channel, n_channel), dtype=dtype)
    for i in range(n_channel):
        kernel[:, :, i, i] = 1.0
    return kernel
    
def identical_weight(n_channel:int=None, dtype:str="float32") -> np.ndarray:
    if n_channel is None:
        return np.array([1.0], dtype=dtype)
    else:
        return np.ones((n_channel,), dtype=dtype)

def identical_bias(n_channel:int=None, dtype:str="float32") -> np.ndarray:
    if n_channel is None:
        return np.array([0.0], dtype=dtype)
    else:
        return np.zeros((n_channel,), dtype=dtype)

def count_node(node, target:list=None):
    cnt = 0
    if target is None:
        for ni in node["layers"]:
            cnt += len(node["layers"][ni])
    else:
        for ni in node["layers"]:
            if ni in target:
                cnt += len(node["layers"][ni])
    return cnt

def use_block_function(idx:int, node:dict) -> bool:
    if idx == 1:
        return _use_scale(node)
    elif idx == 2:
        return node["residual"]
    elif idx == 3:
        return True if node["activation"] is not None else False
    else: # default
        return None






################
##### Concat
################
def concat_to_conv(refer:nx.DiGraph) -> nx.DiGraph:
    graph = nx.DiGraph()
    nodes, edges = od({}), []
    cnt = max([i for i in refer.nodes]) + 1

    # Backdata
    for bi in refer.nodes:
        if refer.nodes[bi]["block_function"] == "Concat":
            _nodes = _concat_to_conv(refer.nodes[bi], cnt)
            for i in range(len(_nodes)):
                nodes[cnt + i] = _nodes[i]
            cnt += len(_nodes)
        else:
            nodes[bi] = refer.nodes[bi]

    # Nodes
    graph.add_nodes_from(nodes.keys())
    graph = update_nodes(graph, nodes)

    # Edges
    graph = update_edges(graph)
    return graph


################
##### GAP
################
def gap_to_conv(refer:nx.DiGraph) -> nx.DiGraph:
    graph = nx.DiGraph()
    nodes, edges = od({}), []
    cnt = max([i for i in refer.nodes]) + 1

    # Backdata
    for bi in refer.nodes:
        if refer.nodes[bi]["block_function"] == "GlobalAveragePool":
            # _nodes = _gap_to_conv(refer.nodes[bi], cnt, k=2)
            _nodes = _gap_to_conv(refer.nodes[bi], cnt)
            for i in range(len(_nodes)):
                nodes[cnt + i] = _nodes[i]
            cnt += len(_nodes)
        else:
            nodes[bi] = refer.nodes[bi]

    # Nodes
    graph.add_nodes_from(nodes.keys())
    graph = update_nodes(graph, nodes)

    # Edge
    graph = update_edges(graph)
    return graph

def gap_to_serial_conv(refer:nx.DiGraph) -> nx.DiGraph:
    graph = nx.DiGraph()
    nodes, edges = od({}), []
    cnt = max([i for i in refer.nodes]) + 1

    # Backdata
    for bi in refer.nodes:
        if refer.nodes[bi]["block_function"] == "GlobalAveragePool":
            _nodes = _gap_to_serial_conv(refer.nodes[bi], cnt)
            for i in range(len(_nodes)):
                nodes[cnt + i] = _nodes[i]
            cnt += len(_nodes)
        else:
            nodes[bi] = refer.nodes[bi]

    # Nodes
    graph.add_nodes_from(nodes.keys())
    graph = update_nodes(graph, nodes)

    # Edge
    graph = update_edges(graph)
    return graph


################
##### Grouped Conv
################
def group_conv(refer:nx.DiGraph) -> nx.DiGraph:
    graph = nx.DiGraph()
    nodes, edges = od({}), []
    cnt = max([i for i in refer.nodes]) + 1

    # Backdata
    for bi in refer.nodes:
        if refer.nodes[bi]["block_function"] == "Conv" or refer.nodes[bi]["block_function"] == "ConvTranspose" or refer.nodes[bi]["block_function"] == "IdenticalConv":
            if _is_group_conv(refer.nodes[bi]):
                _nodes = _group_conv(refer.nodes[bi], cnt)
                for i in range(len(_nodes)):
                    nodes[cnt + i] = _nodes[i]
                cnt += len(_nodes)
            else:
                nodes[bi] = refer.nodes[bi]
        else:
            nodes[bi] = refer.nodes[bi]

    # Nodes
    graph.add_nodes_from(nodes.keys())
    graph = update_nodes(graph, nodes)

    # Edge
    graph = update_edges(graph)
    return graph

################
##### Split Conv
################
def split_conv(refer:nx.DiGraph, buffer_size=4096, multiple=8) -> nx.DiGraph:
    graph = nx.DiGraph()
    nodes, edges = od({}), []
    cnt = max([i for i in refer.nodes]) + 1

    # Backdata
    for bi in refer.nodes:
        if refer.nodes[bi]["block_function"] == "Conv" or refer.nodes[bi]["block_function"] == "ConvTranspose" or refer.nodes[bi]["block_function"] == "IdenticalConv":
            if _over_buffer(refer.nodes[bi], buffer_size):
                _nodes = _split_conv(
                    refer.nodes[bi], 
                    cnt, 
                    buffer_size=buffer_size, 
                    multiple=multiple
                )
                for i in range(len(_nodes)):
                    nodes[cnt + i] = _nodes[i]
                cnt += len(_nodes)
            else:
                nodes[bi] = refer.nodes[bi]
        else:
            nodes[bi] = refer.nodes[bi]

    # Nodes
    graph.add_nodes_from(nodes.keys())
    graph = update_nodes(graph, nodes)

    # Edge
    graph = update_edges(graph)
    return graph

################
##### Split MaxPool
################
def split_maxpool(refer:nx.DiGraph, buffer_size=4096, multiple=8) -> nx.DiGraph:
    graph = nx.DiGraph()
    nodes, edges = od({}), []
    cnt = max([i for i in refer.nodes]) + 1

    # Backdata
    for bi in refer.nodes:
        if refer.nodes[bi]["block_function"] == "MaxPool":
            if _over_buffer(refer.nodes[bi], buffer_size):
                _nodes = _split_maxpool(
                    refer.nodes[bi], 
                    cnt, 
                    buffer_size=buffer_size, 
                    multiple=multiple
                )
                for i in range(len(_nodes)):
                    nodes[cnt + i] = _nodes[i]
                cnt += len(_nodes)
            else:
                nodes[bi] = refer.nodes[bi]
        else:
            nodes[bi] = refer.nodes[bi]

    # Nodes
    graph.add_nodes_from(nodes.keys())
    graph = update_nodes(graph, nodes)

    # Edge
    graph = update_edges(graph)
    return graph


################
##### Fullyconnected to Conv
################
def fc_to_conv(refer:nx.DiGraph) -> nx.DiGraph:
    graph = nx.DiGraph()
    nodes, edges = od({}), []
    cnt = max([i for i in refer.nodes]) + 1

    # Backdata
    for bi in refer.nodes:
        if refer.nodes[bi]["block_function"] == "Gemm" or refer.nodes[bi]["block_function"] == "MatMul":
            nodes[bi] = _fc_to_conv(refer.nodes[bi], cnt)
        else:
            nodes[bi] = refer.nodes[bi]

    # Nodes
    graph.add_nodes_from(nodes.keys())
    graph = update_nodes(graph, nodes)

    # Edge
    graph = update_edges(graph)
    return graph

################
##### Align Data (x4)
################
def _check_align(shape_list:list, m=4) -> list:
    result = []
    for shape in shape_list:
        _, _, w, c = shape
        wr = w % m
        cr = c % m
        if wr == 0 or cr == 0:
            result.append(0)
        else:
            result.append(m - cr if cr > 0 else 0)

    return result 

def _check_inout_align(node:dict, multiple:int=8) -> list:
    i_residual = _check_align(node["input_shapes"], m=multiple)
    o_residual = _check_align(node["output_shapes"], m=multiple)
    aligned = True if np.sum(np.array(i_residual + o_residual, dtype="int32")) == 0 else False
    return aligned, (i_residual, o_residual) 

def _extension_kernel(kernel:np.ndarray, residual:int, axis:int=2) -> np.ndarray:
    h, w, i, o = kernel.shape
    if axis == 2: # i
        kernel_ext = np.zeros((h, w, i + residual, o), kernel.dtype)
        kernel_ext[:, :, :i, :] = kernel
    else: # o
        kernel_ext = np.zeros((h, w, i, o + residual), kernel.dtype)
        kernel_ext[:, :, :, :o] = kernel

    return kernel_ext

def _extension_weight(weight:np.ndarray, residual:int) -> np.ndarray:
    o = weight.size
    weight_ext = np.zeros((o + residual,), weight.dtype)
    weight_ext[:o] = weight
    return weight_ext

def _align_data(
    node:dict, 
    cnt:int, 
    info:list, 
    i_en:bool=True, o_en:bool=True, 
    dtype="float32"
) -> dict: 
    _node = copy.deepcopy(node)
    i_residual, o_residual = info

    if i_en:
        # shapes
        for i, (i_r, i_s) in enumerate(zip(i_residual, node["input_shapes"])):
            _node["input_shapes"][i] = list(i_s)
            _node["input_shapes"][i][-1] += i_r

        # params
        _node["params"][0] = _extension_kernel(_node["params"][0], i_residual[0], axis=2)

    if o_en:
        # shapes: len == 1
        for i, (o_r, o_s) in enumerate(zip(o_residual, node["output_shapes"])):
            _node["output_shapes"][i] = list(o_s)
            _node["output_shapes"][i][-1] += o_r

        # org_shapes: len == 1
        if "org_shapes" in node:
            for i, (o_r, o_o) in enumerate(zip(o_residual, node["org_shapes"])):
                _node["org_shapes"][i] = list(o_o)
                _node["org_shapes"][i][-1] += o_r

        # params
        _node["params"][0] = _extension_kernel(_node["params"][0], o_residual[0], axis=3)
        _node["params"][1] = _extension_weight(_node["params"][1], o_residual[0])
        _node["params"][2] = _extension_weight(_node["params"][2], o_residual[0])

    # channel_index
    if "channel_index" in _node:
        if _node["channel_index"] is not None:
            _node["channel_index"] = list(_node["channel_index"])
            if i_en: _node["channel_index"][1] += i_residual[0]
            if o_en: _node["channel_index"][2] += i_residual[0]

    # flag
    _node["channel_residual"] = (
        i_residual[0] if i_en else 0, 
        o_residual[0] if o_en else 0, 
    )

    return _node

def align_data(refer:nx.DiGraph, multiple=8) -> nx.DiGraph:
    graph = nx.DiGraph()
    nodes, edges = od({}), []
    cnt = max([i for i in refer.nodes]) + 1

    # Backdata
    for bi in refer.nodes:
        aligned, info = _check_inout_align(refer.nodes[bi], multiple=multiple)
        if not aligned:
            nodes[bi] = _align_data(
                refer.nodes[bi], cnt, info, 
                i_en=False if len(search_parents(refer, bi)) == 0 else True, 
                o_en=True
            )
        else:
            nodes[bi] = refer.nodes[bi]

    # Nodes
    graph.add_nodes_from(nodes.keys())
    graph = update_nodes(graph, nodes)

    # Edge
    graph = update_edges(graph)
    return graph

################
##### Reorder
################
def reorder_logit(refer:nx.DiGraph, order=None) -> nx.DiGraph:
    if len(order) == 0:
        return refer

    idxs_node_in    = []
    idxs_node_out   = []
    idxs_node_logit = []
    idxs_edge_logit = []

    for _in, _out in refer.edges:
        idxs_node_in.append(_in)
        idxs_node_out.append(_out)
    set(idxs_node_in)
    set(idxs_node_out)

    for idx in idxs_node_out:
        if idx not in idxs_node_in:
            idxs_node_logit.append(idx)

    assert(len(order) == len(idxs_node_logit)), "The number of logit orders does not match the number of logit blocks."

    new_idx = max(refer.nodes) + 1
    mapping = {}
    for order_idx in order:
        logit_idx = idxs_node_logit[order_idx]
        mapping[logit_idx] = new_idx
        refer = nx.relabel_nodes(refer, mapping) # relabeling logit node(because logit node must be in the back)
        new_idx += 1

    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(refer.nodes(data=True)))
    graph.add_edges_from(refer.edges(data=True))

    return graph

################
##### Merge
################
def merge_block(refer:nx.DiGraph) -> nx.DiGraph:
    diff = None
    while diff is None or diff > 0:
        graph = nx.DiGraph()
        nodes, edges = od({}), []
        cnt = max(refer.nodes) + 1
        block_nodes = []

        for bi in refer.nodes:
            if bi not in block_nodes:
                merge_node, merge_idx = _check_merge_block(bi, refer)
                if merge_node is not None:
                    nodes[bi] = _merge_block_node(
                        refer.nodes[bi], merge_node, bi, 
                    )
                    block_nodes.append(merge_idx)

                else:
                    nodes[bi] = refer.nodes[bi]

        # Nodes
        graph.add_nodes_from(nodes.keys())
        graph = update_nodes(graph, nodes)

        diff = len(refer.nodes) - len(graph.nodes)
        refer = copy.deepcopy(graph)

    # Edge
    graph = update_edges(graph)

    return graph

################
##### Reshape Output
################
def reshape_outputs(refer:nx.DiGraph) -> nx.DiGraph:
    graph = nx.DiGraph()
    nodes, edges = od({}), []
    cnt = max([i for i in refer.nodes]) + 1

    # Backdata
    for bi in refer.nodes:
        shape = _get_reshape(bi, refer)
        if shape is not None:
            nodes[bi] = _reshape_outputs(refer.nodes[bi], shape, cnt)
        else:
            nodes[bi] = refer.nodes[bi]

    # Nodes
    graph.add_nodes_from(nodes.keys())
    graph = update_nodes(graph, nodes)

    # Edge
    graph = update_edges(graph)
    return graph



# if __name__ == '__main__':
#     tm = TachyGraph(onnx.load(src_file))
#     # tm = tm.shrink_graph()
#     # cfg = tm.get_param_config()
#     # print(json.dumps(cfg, indent=4))
