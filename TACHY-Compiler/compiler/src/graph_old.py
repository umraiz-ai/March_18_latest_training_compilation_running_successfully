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
# import json
import importlib
from collections import OrderedDict as od

import numpy as np
np.set_printoptions(threshold=np.inf)
import networkx as nx
# import matplotlib.pyplot as plt

from tachy_layers import Input

import ddesigner_api.numpy.xwn.functions as xwnf


_DEBUG = False


#################################
####### Internal - Methods
#################################
def _filter_all_hit(targets:list, searched:list, edges):
    pre, post = [], []
    searched = set(searched)
    for t in targets:
        if len(set(in_neighbors(edges, t)) - searched) > 0:
            post.append(t)
        else:
            pre.append(t)
    return pre

def _transpose(x, in_order, out_order, mode="dimension"):
    _in_order = {}
    _out_order = {}
    forward_order = []
    for i in range(len(in_order)):
        _in_order[in_order[i]] = i
    for order in out_order:
        forward_order.append(_in_order[order])

    if mode == "dimension":
        return np.transpose(x, forward_order)
    elif mode == "value":
        return np.take(x, forward_order)

def _reverse(x, channels):
    return np.flip(x, axis=channels)

# def _get_exclusive(x:list, y:list) -> list:
#     return sorted(list(set(y) - set(x)))

# def _build_by_input(graph:nx.DiGraph) -> nx.DiGraph:
#     return graph

def _create_layer(
    cfg, 
    inputs, 
    param_dtype, op_dtype,
    default_pad_order:str="down_right", 
    default_pad_mode:str="dynamic", 
    dump_dir=None, 
):
    assert len(inputs) > 0
    cls = getattr(importlib.import_module('tachy_layers'), cfg['op_type'])
    cfg["default_pad_order"] = default_pad_order
    cfg["default_pad_mode"] = default_pad_mode
    layer = cls(
        cfg, 
        inputs, 
        param_dtype, op_dtype,
        dump_dir=dump_dir, 
    )
    return layer

def _get_support_kernel_size(ksize, support_ksize):
    for ks in support_ksize:
        if ks >= ksize:
            return ks
    return None



#################################
####### External - Methods
#################################
def get_root_reaf_nodes(m:nx.DiGraph) -> list:
    edges = np.asarray(m.edges, dtype='uint32') # (N,2)
    in_nodes = set(edges[..., 0])
    out_nodes = set(edges[..., 1])
    roots = list(in_nodes - out_nodes)
    reafs = list(out_nodes - in_nodes)
    return roots, reafs


def connection_by_edge(graph:nx.DiGraph, edge:tuple) -> list:
    no, ni = edge
    ci = graph.nodes[no]["outputs"]
    co = graph.nodes[ni]["inputs"]
    cis = graph.nodes[no]["output_shapes"]
    cos = graph.nodes[ni]["input_shapes"]

    names, shapes = [], []
    for i, c in enumerate(ci):
        if c in co:
            names.append(c)
            shapes.append(cis[i])

    return names, shapes

def index_by_name(graph:nx.DiGraph, name:str) -> int:
    index = None
    for ni in graph.nodes:
        if graph.nodes[ni]["name"] == name:
            index = ni
            break

    return index


def search_children(graph:nx.DiGraph, idx:int) -> list:
    children = []
    outputs = graph.nodes[idx]["outputs"]
    for o_name in outputs:
        for bi in graph.nodes:
            inputs = graph.nodes[bi]["inputs"]
            if o_name in inputs:
                children.append(bi)
                # break

    return children

def search_parents(graph:nx.DiGraph, idx:int) -> list:
    parents = []
    inputs = graph.nodes[idx]["inputs"]
    for i_name in inputs:
        for bi in graph.nodes:
            outputs = graph.nodes[bi]["outputs"]
            if i_name in outputs:
                parents.append(bi)
                # break

    return parents


def get_connected(idx:int, dg:nx.DiGraph, targets:list) -> list:
    result = []
    for oi in list(dg.neighbors(idx)):
        if oi in targets: 
            result.append(oi)
        else:
            # result.append(get_connected(oi, dg, targets))
            result += get_connected(oi, dg, targets)

    return result

def in_neighbors(edges, idx:int) -> list:
    result = []
    for e in edges:
        if e[1] == idx:
            result.append(e[0])
    return result

def out_neighbors(edges, idx:int) -> list:
    result = []
    for e in edges:
        if e[0] == idx:
            result.append(e[1])
    return result

def in_edges(edges, idx:int) -> list:
    result = []
    i_nodes = in_neighbors(edges, idx)
    for n in i_nodes:
        result.append((n, idx,))
    return result

def out_edges(edges, idx:int) -> list:
    result = []
    o_nodes = out_neighbors(edges, idx)
    for n in o_nodes:
        result.append((idx, n,))

    return result

def get_edges(edges, idx:int) -> list:
    result = []
    i_nodes = in_neighbors(edges, idx)
    o_nodes = out_neighbors(edges, idx)
    for n in i_nodes:
        result.append((n, idx,))
    for n in o_nodes:
        result.append((idx, n,))

    return result

def update_edges(graph:nx.DiGraph) -> nx.DiGraph:
    edges = []
    for ni in graph.nodes:
        for oi in search_children(graph, ni):
            edges.append((ni, oi,))
    graph.add_edges_from(edges)
    return graph

def update_nodes(graph:nx.DiGraph, values:dict) -> nx.DiGraph:
    for ni, nv in graph.nodes.items():
        nv.update(values[ni])
    return graph

def sort_graph(graph:nx.DiGraph, edges):
    # Sort Nodes
    roots, _ = get_root_reaf_nodes(graph)
    targets = []
    for r in roots: targets += out_neighbors(edges, r)
    searched = roots
    in_targets, out_targets = [], []
    while len(targets) > 0:
        targets = _filter_all_hit(targets, searched, edges)
        for t in targets:
            searched.append(t)

        for t in targets:
            in_targets += in_neighbors(edges, t)
            in_targets = list(set(in_targets) - set(searched))

        for t in targets:
            out_targets += out_neighbors(edges, t)
            out_targets = list(set(out_targets) - set(searched))

        if len(in_targets) == 0:
            targets = out_targets
        else:
            targets = in_targets

    assert len(searched) == len(graph)

    dg_sorted = nx.DiGraph()
    dg_sorted.add_nodes_from(searched)
    dg_sorted.add_edges_from(edges)

    return dg_sorted 

#####################
########## Parameters
#####################
def get_square_kernel(x):
    if x.ndim != 4:
        return x
    else:
        kh, kw, _, _ = x.shape
        if kh == kw:
            return x
        else:
            if kh < kw:
                assert kh == 1
                x = np.tile(x, (kw, 1, 1, 1))
            else:
                assert kw == 1
                x = np.tile(x, (1, kh, 1, 1))
            return x

def get_extension_kernel(x, weight=None, support_ksize:tuple=(1, 3, 5, 7)):
    if x.ndim != 4:
        return x, np.ones(x.shape, dtype="bool")
    else:
        h, w, i, o = x.shape
        ks = _get_support_kernel_size(max(h, w), support_ksize)

        if weight is None: np.ones((1,), dtype=x.dtype)
        k = np.ones((ks, ks, i, o), dtype=x.dtype) * weight
        k[:h, ks - w:, ...] = x

        v = np.zeros((ks, ks, i, o), dtype="bool")
        v[:h, ks - w:, ...] = True
        return k, v

def optimize_precision(
    graph:nx.DiGraph, 
    # limit_top:float=1e-1,
    limit_top:float=1e-0,
    limit_bottom:float=1e-3,
) -> nx.DiGraph:
    for ni in graph.nodes:
        node = graph.nodes[ni]
        k = node["params"][0]
        w = node["params"][1]
        kavg = np.mean(np.abs(k), axis=(0, 1, 2))
        kw = np.where(kavg == 0, limit_top, kavg)
        kw = np.where(kw > limit_top, 1.0, limit_top / kw)  # Down scaling
        # kw = np.where(w / kw < limit_bottom, 1.0, kw)       # Up scaling
        kw = kw.astype("float16").astype("float32")
        node["params_scale_factor"] = kw

        # print(node["name"], kw.shape, np.max(kw), np.min(kw))
        # _kavg = np.mean(np.abs(k * kw), axis=(0, 1, 2)) / kw
        # print(node["name"], kavg.shape, np.max(kavg), np.min(kavg))
        # print(node["name"], _kavg.shape, np.max(_kavg), np.min(_kavg))

        # node["params"][0] *= kw
        # node["params"][1] /= kw

    return graph

def invert_input_channel(
    graph:nx.DiGraph, 
) -> nx.DiGraph:
    for ni in graph.nodes:
        in_idxs = search_parents(graph, ni)
        if len(in_idxs) == 0: 
            node = graph.nodes[ni]
            node["params"][0] = node["params"][0][:, :, ::-1, :]

    return graph


def convert_param(
    graph:nx.DiGraph, 
    cvt_cfg:dict
) -> nx.DiGraph:
    for i, n_idx in enumerate(graph.nodes):
        op_type = graph.nodes[n_idx]['op_type']
        # op_dim = graph.nodes[n_idx]['configs']['op_dim'] if 'op_dim' in graph.nodes[n_idx]['configs'] else ''
        # cfg_key = '{}{}'.format(op_type, op_dim)
        cfg_key = '{}'.format(op_type)

        if cfg_key in cvt_cfg:
            result = []
            for x, cfg in zip(graph.nodes[n_idx]['params'], cvt_cfg[cfg_key]):
                # Align
                rev             = cfg['REVERSE']        if 'REVERSE'        in cfg.keys() else False
                order           = cfg['ORDER']          if 'ORDER'          in cfg.keys() else False
                expand          = cfg['EXPAND']         if 'EXPAND'         in cfg.keys() else False
                bn              = cfg['BATCHNORMAL']    if 'BATCHNORMAL'    in cfg.keys() else False
                # tc                  = cfg['TCONV']      if 'TCONV'      in cfg.keys() else False

                # Order & Inverse
                if rev:
                    x = _reverse(x, rev)

                if order:
                    in_order, out_order = order
                    # print(x.shape, in_order, out_order)
                    x = _transpose(x, in_order, out_order)

                if expand:
                    for i in expand:
                        x = np.expand_dims(x, axis=i)

                if bn:
                    if x.ndim == 1:
                        x = x[None, None, None, :]
                    elif x.ndim == 4:
                        x = np.transpose(x, (0,2,3,1))
                    else:
                        x

                result.append(x)

            # graph.nodes[n_idx]['params_convert'] = result
            graph.nodes[n_idx]['params'] = result

    return graph

def convert_shape(
    graph:nx.DiGraph, 
    shp_cfg:dict
) -> nx.DiGraph:
    for i, n_idx in enumerate(graph.nodes):
        op_type = graph.nodes[n_idx]['op_type']
        if op_type in shp_cfg:
            result = []
            for x, order in zip(graph.nodes[n_idx]['params'], shp_cfg[op_type]):
                assert len(x) == 4
                in_order, out_order = order["ORDER"]
                s = _transpose(x, in_order, out_order, mode="value")
                # s = np.zeros_like(x)
                # s[0] = x[0]
                # s[1] = x[2]
                # s[2] = x[3]
                # s[3] = x[1]
                result.append(s)

            graph.nodes[n_idx]['params'] = result

    return graph


def _expand_dim(x:np.ndarray, axis=None) -> np.ndarray:
    assert x.ndim != 2
    if x.ndim == 3:
        if axis == "h": # (I,O,W)->(I,O,H,W) / (B,C,W)->(B,C,H,W)
            return np.expand_dims(x, 2) 
        elif axis == "w":
            return np.expand_dims(x, 3) 

    elif x.ndim == 1:
        return x[None, :, None, None] 
    else: # x.dim == 4 
        return x

def _expand_configs(configs:dict, axis=None):
    if "kernel_shape" in configs: 
        if axis == "h":
            configs["kernel_shape"] = [1] + configs["kernel_shape"]
        elif axis == "w":
            configs["kernel_shape"] = configs["kernel_shape"] + [1]

    if "pads" in configs: 
        if axis == "h":
            configs["pads"] = [0, configs["pads"][0], 0, configs["pads"][1]]
        elif axis == "w":
            configs["pads"] = [configs["pads"][0], 0, configs["pads"][1], 0]

    if "strides" in configs: 
        if axis == "h":
            configs["strides"] = [1] + configs["strides"]
        elif axis == "w":
            configs["strides"] = configs["strides"] + [1]

    if "dilations" in configs: 
        if axis == "h":
            configs["dilations"] = [1] + configs["dilations"]
        elif axis == "w":
            configs["dilations"] = configs["dilations"] + [1]

    # configs["op_dim"] = 2
    return configs


def expand_param(
    graph:nx.DiGraph, 
    axis=None,
) -> nx.DiGraph:
    for i, n_idx in enumerate(graph.nodes):
        if 'op_dim' in graph.nodes[n_idx]['configs']:
            # if graph.nodes[n_idx]['configs']['op_dim'] == 1 and len(graph.nodes[n_idx]['params']) > 0:
            if graph.nodes[n_idx]['configs']['op_dim'] == 1:
                if len(graph.nodes[n_idx]['params']) > 0:
                    params = []
                    for i, p in enumerate(graph.nodes[n_idx]['params']):
                        params.append(_expand_dim(p, axis=axis))

                    graph.nodes[n_idx]['params'] = params

                graph.nodes[n_idx]['configs'] = _expand_configs(
                    graph.nodes[n_idx]['configs'], axis=axis
                )

    return graph



def _get_bins(
    target_bit:int,
    target_max:float,
    dtype:str="float32",
) -> list:
    result = []
    for i in range(2 ** (target_bit - 1)):
        result.append(target_max / (2 ** i))

    result_rev = result[0:]
    result_rev.reverse()
    result = result + [0.0] + result_rev
    return np.array(result, dtype=dtype)
        

def optimize_param(
    graph:nx.DiGraph, 
    ref_graph:nx.DiGraph, 
    opt_cfg:dict,
    use_dim:bool=True,
    target_bit:int=4,
    target_max:float=4.0,
    verbose:bool=False,
) -> nx.DiGraph:

    val_total_size = 0
    val_total_nonzeros = 0
    inval_total_size = 0
    inval_total_nonzeros = 0

    for i, ni in enumerate(graph.nodes):
        node = graph.nodes[ni] 
        ref_node = ref_graph.nodes[ni] 
        op_type = node['op_type']
        if use_dim:
            op_dim = node['configs']['op_dim'] if 'op_dim' in node['configs'] else ''
            cfg_key = '{}{}'.format(op_type, op_dim)
        else:
            cfg_key = '{}'.format(op_type)

        fn_val = False if node["block_function"] == "IdenticalConv" or node["block_function"] == "SerialConv" else True
        # TODO: BS3_Conv, BS3_Gemm, BS3_MaxPool
        if cfg_key in opt_cfg:
            x = node['params'][0]
            r = ref_node['params'][0]

            m = xwnf.get_magnitude(x)
            rm = xwnf.get_magnitude(r)

            # # NxM
            # x = get_square_kernel(x)
            # # x_valid = np.ones_like(x, dtype="bool")
            # r = get_square_kernel(r)

            x, x_valid = get_extension_kernel(x, weight=m)
            r, _ = get_extension_kernel(r, weight=np.max(r, axis=(0, 1)))

            residual = np.where(m == 0., 0., r / m)
            residual = np.where(x_valid, residual, np.max(residual))
            residual = xwnf.quantization(residual, _get_bins(target_bit, target_max))
            bit, max_scale, _ = xwnf.find_bit_scale(residual)

            if bit <= 2 ** target_bit and max_scale <= target_max:
                # Adjust pruning
                m = np.where(rm == 0., 0., m)

                total = float(np.size(m))
                non_zeros = np.count_nonzero(m)

                if fn_val:
                    val_total_size += total
                    val_total_nonzeros += non_zeros
                else:
                    inval_total_size += total
                    inval_total_nonzeros += non_zeros

                if verbose:
                    print("[INFO]: {}, bit={} -> {}, max_scale={} -> {}, pruning ratio({})= {} / {} = {}".format(
                        node["name"], 
                        max(np.log2(bit), 1.), 
                        float(target_bit), 
                        max_scale, 
                        target_max,
                        fn_val,
                        total - non_zeros,
                        total, 
                        (total - non_zeros) / total * 100.
                    ))

                bit = target_bit
                max_scale = target_max

                _, scl = xwnf.get_scale(x, bit, max_scale)
                scl = scl[..., 0]

                scale_factor = node["params_scale_factor"] if "params_scale_factor" in node else np.array([1], dtype="float32")
                node['params_sign'] = xwnf.get_sign(x)
                node['params_magnitude'] = m * scale_factor
                node['params_scale'] = scl
                node['params_header'] = xwnf.get_header(m, use_pruning=True)

                node['params_bit'] = bit
                node['params_shape'] = x.shape
                node['params_max_scale'] = max_scale
                
                node["params"][1] /= scale_factor

            else: # FC
                print("[INFO]: {}, bit={} -> {}, max_scale={} -> {}".format(node["name"], bit, "x", max_scale, "x"))
                node['params_header'] = xwnf.get_header(x)

    print("[INFO]: {}, training pruning ratio= {} / {} = {}".format(
        "Total", 
        val_total_size - val_total_nonzeros, 
        val_total_size, 
        (val_total_size - val_total_nonzeros) / val_total_size * 100.)
    )
    print("[INFO]: {}, dummy pruning ratio= {} / {} = {}".format(
        "Total", 
        inval_total_size - inval_total_nonzeros, 
        inval_total_size, 
        (inval_total_size - inval_total_nonzeros) / inval_total_size * 100.)
    )


    return graph


#####################
########## Inference
#####################

def build(
    graph:nx.DiGraph, 
    param_dtype:str='float32', 
    op_dtype:str='float32', 
    default_pad_order:str="down_right", 
    default_pad_mode:str="dynamic", 
    dump_dir:str=None
):
    inputs = od({})
    layers = od({})
    outputs = []
    for i in graph.nodes:
        layer_name = graph.nodes[i]['name']
        # Register inputs
        # in_idxs = in_neighbors(graph.edges, i) # without order
        in_idxs = search_parents(graph, i)
        if len(in_idxs) == 0: 
            in_name = 'input_{}'.format(i)
            inputs[in_name] = Input(
                in_name, 
                param_dtype, op_dtype,
                dump_dir=dump_dir, 
            )
            in_names = [in_name]
        else:
            in_names = [graph.nodes[j]['name'] for j in in_idxs]

        # Register outputs
        # out_idxs = out_neighbors(graph.edges, i)
        out_idxs = search_children(graph, i)
        if len(out_idxs) == 0: 
            outputs.append(layer_name)

        # Create & Register layer
        layer = _create_layer(
            graph.nodes[i], 
            in_names, 
            param_dtype, op_dtype,
            default_pad_order=default_pad_order, 
            default_pad_mode=default_pad_mode, 
            dump_dir=dump_dir,
        )
        layers[layer_name] = layer

    return inputs, layers, outputs

def feed_inputs(inputs, source):
    for key, layer in inputs.items():
        layer.forward(source)
    return True

def feed_layers(inputs, layers):
    for key, layer in layers.items():
        x = get_output(layer.inputs, inputs, layers)
        layer.forward(x)
    return True

def get_output(outputs, inputs, layers, crop=False):
    result = []
    for l_name in outputs:
        l = None
        if l_name in layers: l = layers[l_name]
        elif l_name in inputs: l = inputs[l_name]

        if l is not None: result += l.get(crop)

    return result

def update_config(
    graph:nx.DiGraph,
    input_shape:tuple,
    default_pad_order:str="down_right", 
    default_pad_mode:str="dynamic", 
): # TODO: input, result
    inputs, layers, outputs = build(
        graph, 
        default_pad_order=default_pad_order,
        default_pad_mode=default_pad_mode,
    )
    sample = np.random.normal(size=input_shape)
    feed_inputs(inputs, sample)
    feed_layers(inputs, layers)
    for li, ni in zip(layers, graph.nodes):
        graph.nodes[ni]["input_shapes"] = layers[li].get_input_shapes()
        graph.nodes[ni]["output_shapes"] = layers[li].get_output_shapes()
        graph.nodes[ni]["pads_dynamic"] = layers[li].get_pads_dynamic()
    return graph

# if __name__ == '__main__':
#     tm = TachyGraph(onnx.load(src_file))
#     # tm = tm.shrink_graph()
#     # cfg = tm.get_param_config()
#     # print(json.dumps(cfg, indent=4))
