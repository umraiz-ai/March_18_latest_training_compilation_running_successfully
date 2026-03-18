import os, sys
import numpy as np
import networkx as nx
from graph import in_neighbors, out_neighbors, search_children, search_parents

def get_bi(nodes:dict, bi:int):
    return int(nodes[bi]["name"].split('_')[-1])

def get_operation(nodes:dict, bi:int):
    block_fn = nodes[bi]["block_function"]
    if block_fn == "Conv" or \
       block_fn == "IdenticalConv" or \
       block_fn == "ConvSplit" or \
       block_fn == "SerialConv" or \
       block_fn == "BatchNormalization":
        return "Conv"
    elif block_fn == "ConvTranspose" or \
         block_fn == "ConvTransposeSplit":
        return "TransposeConv"
    elif "MaxPool" in block_fn:
        return "MaxPool"
    elif "Gemm" in block_fn:
        return "Gemm"
    else:
        raise Exception("Unknown block_function : {}".format(block_fn))

def get_input_shape(nodes:dict, bi:int):
    shape = list(nodes[bi]["input_shapes"][0])

    if len(shape) == 2:   shape = [1,] + shape # Gemm
    elif len(shape) == 4: shape = shape[1:] # B, H, W, D
    else:                 raise Exception("Unknown input shape dimension : {}".format(shape))

    return shape

def get_output_shape(nodes:dict, bi:int):
    # shape = list(nodes[bi]["output_shapes"][0]) if "op_shapes" not in nodes[bi] else nodes[bi]["op_shapes"][0]
    # print(nodes[bi]["org_shapes"])
    # shape = list(nodes[bi]["org_shapes"][0]) if "op_shapes" not in nodes[bi] else nodes[bi]["op_shapes"][0]
    if "op_shapes" in nodes[bi]:
        shape = nodes[bi]["op_shapes"][0]
    elif "org_shapes" in nodes[bi]:
        shape = list(nodes[bi]["org_shapes"][0])
    else:
        shape = list(nodes[bi]["output_shapes"][0])

    if len(shape) == 2:   shape = [1,] + shape # Gemm
    elif len(shape) == 4: shape = shape[1:] # B, H, W, D
    else:                 raise Exception("Unknown output shape dimension : {}".format(shape))

    return shape

def get_kernel_shape(nodes:dict, bi:int):
    param, _, __ = nodes[bi]["params"]
    dummy_in_d, dummy_out_d = nodes[bi]["channel_residual"]

    if len(param.shape) == 2:   param = param[None,None,:] # Gemm

    if get_operation(nodes, bi) == "MaxPool":
        ker_h, ker_w = nodes[bi]["configs"]["kernel_shape"][:2]
        ker_in_d = param.shape[2]
        ker_out_d = param.shape[3]
    else:
        ker_h = param.shape[0]
        ker_w = param.shape[1]
        ker_in_d = param.shape[2]
        ker_out_d = param.shape[3] - dummy_out_d

    return [ker_h, ker_w, ker_in_d, ker_out_d]

def get_stride_shape(nodes:dict, bi:int):
    return list(nodes[bi]["configs"]["strides"])

# return t, l, b, r
def get_padding_shape(nodes:dict, bi:int):
    pad = nodes[bi]["pads_dynamic"]
    p_t, p_b = pad[0]
    p_l, p_r = pad[1]
    p_s = 0

    return [p_t, p_l, p_b, p_r, p_s]

def get_activation(nodes:dict, bi:int):
    activation = nodes[bi]["activation"]
    if activation is not None:
        cfg = {}
        if activation == 0.0:     cfg["name"] = "Relu"
        elif activation == 0.125: cfg["name"] = "LeakyRelu"
        else:                     raise Exception("[{}] Unknown Relu activation : {}".format(nodes[bi]["name"], activation))
        return cfg
    else:
        return None

def get_residual_with(nodes:dict, bi:int):
    if nodes[bi]['residual']:
        resi_from = nodes[bi]["inputs"][-1]

        for idx, _bi in enumerate(nodes):
            if resi_from in nodes[_bi]["outputs"]:
                return idx
    return None

def get_input_from(graph:nx.DiGraph, bi:int):
    ret = []
    # in_neighbor = in_neighbors(graph.edges, bi)
    in_neighbor = search_parents(graph, bi)

    if len(in_neighbor) == 0:
        return -1

    for idx, _bi in enumerate(graph.nodes):
        if _bi == in_neighbor[0]:
            return idx

    raise Exception('Unable to find input_from')

def get_output_tos(graph:nx.DiGraph, bi:int):
    ret = []
    # out_neighbor = out_neighbors(graph.edges, bi)
    out_neighbor = search_children(graph, bi)
    for idx, _bi in enumerate(graph.nodes):
        if _bi in out_neighbor:
            ret.append(idx)

    return ret

def get_xwn_bit(nodes:dict, bi:int):
    block_function = nodes[bi]["block_function"]
    if block_function == "MaxPool" or block_function == "Gemm":
        return 1
    else:
        return nodes[bi]["params_bit"]

def get_xwn_scale(nodes:dict, bi:int):
    block_function = nodes[bi]["block_function"]
    if block_function == "MaxPool" or block_function == "Gemm":
        return 1
    else:
        return nodes[bi]["params_max_scale"]

def get_param_size(nodes:dict, bi:int):
    return nodes[bi]["params_size"]

def get_is_start(nodes:dict, bi:int):
    if nodes[bi]["inputs"][0] == "input":
        return True
    else:
        return False

def get_is_logit(graph:nx.DiGraph, bi:int):
    output_tos = get_output_tos(graph, bi)

    if len(output_tos) == 0:
        return True
    else:
        return False

def get_first_depth(nodes:dict, bi:int):
    channel_index = nodes[bi]["channel_index"]
    if channel_index is None: return 0
    else: return channel_index[0]

def get_channel_idx(nodes:dict, bi:int):
    channel_index = nodes[bi]["channel_index"]
    return channel_index

def get_channel_residual(nodes:dict, bi:int):
    channel_residual = nodes[bi]["channel_residual"]
    return channel_residual

def get_op_dim(nodes:dict, bi:int):
    return nodes[bi]["configs"]["op_dim"]
