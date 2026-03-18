#coding:utf-8

import os

def insert_taps(s:str, n:int=1) -> str:
    taps = "".join(["    " for i in range(n)])
    return taps + s
    

def _get_tops(info:list) -> list:
    result = []
    for t in info:
        result.append(insert_taps("top: \"{}\"\n".format(t), n=1))
    return result

def _get_bottoms(info:list) -> list:
    result = []
    for b in info:
        result.append(insert_taps("bottom: \"{}\"\n".format(b), n=1))
    return result

def _get_pads(info:list) -> list:
    result = []
    if len(info) == 0:
        result = [0, 0, 0, 0]
    elif len(info) == 2:
        result = [info[0], info[1], 0, 0]
    elif len(info) == 4:
        result = [info[0], info[2], info[1], info[3]]

    return result

def _get_kernel_size(info:list) -> list:
    result = []
    if len(info) == 1:
        result = [info[0], 1]
    elif len(info) == 2:
        result = [info[0], info[1]]

    return result

def _get_strides(info:list) -> list:
    result = []
    if len(info) == 1:
        result = [info[0], 1]
    elif len(info) == 2:
        result = [info[0], info[1]]

    return result


###############################
####### Custom Line Gen.
###############################
def get_conv_lines(info:dict) -> list:
    '''
    args
        "NAME":{
            "SHAPE":tuple, 
        }
    '''
    bottoms = _get_bottoms(info['bottoms'])
    tops = _get_tops(info['tops'])
    pad_t, pad_b, pad_l, pad_r = _get_pads(info['pads'])
    kernel_h, kernel_w = _get_kernel_size(info['kernel_size'])
    stride_h, stride_w = _get_strides(info['strides'])

    return [
        "############ {} Block Auto Generated.\n".format(info['name']),
        "layer {\n",
        insert_taps("name: \"{}\"\n".format(info['name']), n=1),
        insert_taps("type: \"{}\"\n".format('Convolution'), n=1),
        "\n"
    ] + bottoms + tops + [
        "\n",
        insert_taps("convolution_param {\n", n=1),
        # insert_taps("num_output: {}\n".format(info['shape'][0]), n=2),
        insert_taps("num_output: {}\n".format(info['shape'][-1]), n=2),
        insert_taps("bias_term: {}\n".format('false' if not info['bias'] else 'true'), n=2),
        insert_taps("stride_h: {}\n".format(stride_h), n=2),
        insert_taps("stride_w: {}\n".format(stride_w), n=2),
        insert_taps("kernel_h: {}\n".format(kernel_h), n=2),
        insert_taps("kernel_w: {}\n".format(kernel_w), n=2),
        insert_taps("pad_t: {}\n".format(pad_t), n=2),
        insert_taps("pad_l: {}\n".format(pad_l), n=2),
        insert_taps("pad_b: {}\n".format(pad_b), n=2),
        insert_taps("pad_r: {}\n".format(pad_r), n=2),
        # insert_taps("kernel_size: {}\n".format(info['kernal_size'][0]), n=2),
        # insert_taps("stride: {}\n".format(info['strides'][0]), n=2),
        # insert_taps("pad_mode: \"{}\" # TODO: Check\n".format('same'), n=2),
        # insert_taps("pad: \"{}\"\n".format(info['pads'][0]), n=2),
        insert_taps("}\n", n=1),
        "}\n",
        "\n",
    ]

# def get_tconv_lines(info:dict, block_name:str):
#     '''
#     args
#         "NAME":{
#             "SHAPE":tuple, 
#         }
#     '''
#     return [
#         "############ {} Block TODO: Check\n".format(block_name),
#         "layer {\n",
#         insert_taps("name: \"{}\"\n".format(block_name), n=1),
#         insert_taps("type: \"{}\"\n".format('TransposeConvolution'), n=1),
#         "\n",
#         insert_taps("bottom: \"{}\" # TODO: Check\n".format(''), n=1),
#         insert_taps("top: \"{}\"\n".format(block_name), n=1),
#         "\n",
#         insert_taps("transpose_convolution_param {\n", n=1),
#         insert_taps("num_output: {}\n".format(info['SHAPE'][3]), n=2),
#         insert_taps("kernel_size: {}\n".format(info['SHAPE'][0]), n=2),
#         insert_taps("stride: {}\n".format('2'), n=2),
#         insert_taps("pad_mode: \"{}\"\n".format('same'), n=2),
#         insert_taps("bias_term: {}\n".format('false' if not info['BIAS'] else 'true'), n=2),
#         insert_taps("}\n", n=1),
#         "}\n",
#         "\n",
#     ]
#     
# def get_bn_lines(info:dict, block_name:str):
#     '''
#     args
#         "NAME":{
#             "SHAPE":tuple, 
#         }
#     '''
#     return [
#         "layer {\n",
#         insert_taps("name: \"{}_bn\"\n".format(block_name), n=1),
#         insert_taps("type: \"{}\"\n".format('BatchNorm'), n=1),
#         "\n",
#         insert_taps("bottom: \"{}\"\n".format(block_name), n=1),
#         insert_taps("top: \"{}\"\n".format(block_name), n=1),
#         "\n",
#         insert_taps("batch_norm_param {\n", n=1),
#         insert_taps("use_global_stats: {}\n".format('true'), n=2),
#         insert_taps("}\n", n=1),
#         "}\n",
#         "\n",
#     ]
    
def get_pooling_lines(info:dict) -> list:
    '''
    args
        "name":...
        "tops":...
        "bottoms":...
        "alpha":...
    '''
    bottoms = _get_bottoms(info['bottoms'])
    tops = _get_tops(info['tops'])
    pad_t, pad_b, pad_l, pad_r = _get_pads(info['pads'])
    kernel_h, kernel_w = _get_kernel_size(info['kernel_size'])
    stride_h, stride_w = _get_strides(info['strides'])

    return [
        "layer {\n",
        insert_taps("name: \"{}\"\n".format(info['name']), n=1),
        insert_taps("type: \"{}\"\n".format('Pooling'), n=1),
        "\n"
    ] + bottoms + tops + [
        "\n",
        insert_taps("pooling_param {\n", n=1),
        insert_taps("pool: {}\n".format('MAX'), n=2),
        insert_taps("stride_h: {}\n".format(stride_h), n=2),
        insert_taps("stride_w: {}\n".format(stride_w), n=2),
        insert_taps("kernel_h: {}\n".format(kernel_h), n=2),
        insert_taps("kernel_w: {}\n".format(kernel_w), n=2),
        insert_taps("pad_t: {}\n".format(pad_t), n=2),
        insert_taps("pad_l: {}\n".format(pad_l), n=2),
        insert_taps("pad_b: {}\n".format(pad_b), n=2),
        insert_taps("pad_r: {}\n".format(pad_r), n=2),
        insert_taps("}\n", n=1),
        "}\n",
        "\n",
    ]

def get_gap_lines(info:dict) -> list:
    '''
    args
        "name":...
        "tops":...
        "bottoms":...
        "alpha":...
    '''
    bottoms = _get_bottoms(info['bottoms'])
    tops = _get_tops(info['tops'])

    return [
        "layer {\n",
        insert_taps("name: \"{}\"\n".format(info['name']), n=1),
        insert_taps("type: \"{}\"\n".format('Pooling'), n=1),
        "\n"
    ] + bottoms + tops + [
        "\n",
        insert_taps("pooling_param {\n", n=1),
        insert_taps("kernel_size: {} # TODO: Check\n".format('0'), n=2),
        insert_taps("stride: {}\n".format(1), n=2),
        insert_taps("pool: {}\n".format('AVE'), n=2),
        # insert_taps("pad_mode: \"{}\" # TODO: Check\n".format('valid'), n=2),
        insert_taps("}\n", n=1),
        "}\n",
        "\n",
    ]
def get_act_lines(info:dict) -> list:
    '''
    args
        "name":...
        "tops":...
        "bottoms":...
        "alpha":...
    '''
    bottoms = _get_bottoms(info['bottoms'])
    tops = _get_tops(info['tops'])

    return [
        "layer {\n",
        insert_taps("name: \"{}\"\n".format(info['name']), n=1),
        insert_taps("type: \"{}\"\n".format('ReLU'), n=1),
        "\n"
    ] + bottoms + tops + [
        "\n",
        insert_taps("relu_param {\n", n=1),
        insert_taps("negative_slope: {}\n".format(info['alpha']), n=2),
        insert_taps("}\n", n=1),
        "}\n",
        "\n",
    ]

def get_fc_lines(info:dict) -> list:
    '''
    args
        "name":...
        "tops":...
        "bottoms":...
        "shape":...
    '''
    bottoms = _get_bottoms(info['bottoms'])
    tops = _get_tops(info['tops'])

    return [
        "layer {\n",
        insert_taps("name: \"{}\"\n".format(info['name']), n=1),
        insert_taps("type: \"{}\"\n".format('InnerProduct'), n=1),
        "\n"
    ] + bottoms + tops + [
        "\n",
        insert_taps("inner_product_param {\n", n=1),
        # insert_taps("num_output: {}\n".format(info['shape'][0]), n=2),
        insert_taps("num_output: {}\n".format(info['shape'][-1]), n=2),
        insert_taps("bias_term: {}\n".format('false' if not info['bias'] else 'true'), n=2),
        insert_taps("}\n", n=1),
        "}\n",
        "\n",
    ]

def get_add_lines(info:dict) -> list:
    '''
    args
        "name":...
        "tops":...
        "bottoms":...
    '''
    bottoms = _get_bottoms(info['bottoms'])
    tops = _get_tops(info['tops'])

    return [
        "layer {\n",
        insert_taps("name: \"{}\"\n".format(info['name']), n=1),
        insert_taps("type: \"{}\"\n".format('Eltwise'), n=1),
        "\n"
    ] + bottoms + tops + [
        "}\n",
        "\n",
    ]

def get_concat_lines(info:dict) -> list:
    '''
    args
        "name":...
        "tops":...
        "bottoms":...
    '''
    bottoms = _get_bottoms(info['bottoms'])
    tops = _get_tops(info['tops'])

    return [
        "layer {\n",
        insert_taps("name: \"{}\"\n".format(info['name']), n=1),
        insert_taps("type: \"{}\"\n".format('Concat'), n=1),
        "\n"
    ] + bottoms + tops + [
        "}\n",
        "\n",
    ]

def get_bn_lines(info:dict) -> list:
    '''
    args
        "name":...
        "tops":...
        "bottoms":...
    '''
    bottoms = _get_bottoms(info['bottoms'])
    tops = _get_tops(info['tops'])

    return [
        "layer {\n",
        insert_taps("name: \"{}\"\n".format(info['name']), n=1),
        insert_taps("type: \"{}\"\n".format('BatchNorm'), n=1),
        "\n"
    ] + bottoms + tops + [
        "\n",
        insert_taps("batch_norm_param {\n", n=1),
        insert_taps("use_global_stats: {}\n".format('true'), n=2),
        insert_taps("}\n", n=1),
        "}\n",
        "\n",
    ]

def get_reshape_lines(info:dict) -> list:
    '''
    args
        "name":...
        "tops":...
        "bottoms":...
    '''
    bottoms = _get_bottoms(info['bottoms'])
    tops = _get_tops(info['tops'])

    return [
        "layer {\n",
        insert_taps("name: \"{}\"\n".format(info['name']), n=1),
        insert_taps("type: \"{}\"\n".format('Reshape'), n=1),
        insert_taps("dims: \"{}\"\n".format(info['shape']), n=1),
        "\n"
    ] + bottoms + tops + [
        "}\n",
        "\n",
    ]

def get_bs3_lines(info:dict) -> list:
    '''
    args
        "name":...
        "tops":...
        "bottoms":...
    '''
    bottoms = _get_bottoms(info['bottoms'])
    tops = _get_tops(info['tops'])

    return [
        "layer {\n",
        insert_taps("name: \"{}\"\n".format(info['name']), n=1),
        insert_taps("type: \"{}\"\n".format('BS3'), n=1),
        "\n"
    ] + bottoms + tops + [
        "}\n",
        "\n",
    ]

