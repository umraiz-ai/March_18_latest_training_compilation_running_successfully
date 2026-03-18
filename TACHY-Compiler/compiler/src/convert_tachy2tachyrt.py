import os, sys
import math
import numpy as np

def set_index(blocks_config:dict, inst:np.ndarray, idx:int):
    idx = blocks_config["blocks"][idx]["index"]

    inst[0] &= 0x00FFFFFF
    inst[0] |= ((idx & 0xFF) << 24)

def set_mode(blocks_config:dict, inst:np.ndarray, idx:int):
    mode = blocks_config["blocks"][idx]["operation"]

    inst[0] &= 0xFFFEFFFC
    if "Conv" == mode:
        inst[0] |= 0x00000000
    elif "TransposeConv" == mode:
        inst[0] |= 0x00010002
    elif "MaxPool" == mode:
        inst[0] |= 0x00010000
    elif "Gemm" == mode:
        inst[0] |= 0x00000001
    elif "Scaler" == mode:
        inst[0] |= 0x00000003
    else:
        raise Exception("Unknown mode : {}".format(mode))

def set_xwn(blocks_config:dict, inst:np.ndarray, idx:int):
    xwn_bit = 1 if (blocks_config["blocks"][idx]["xwn_bit"] == 4) else 0
    xwn_scale = int(math.log2(blocks_config["blocks"][idx]["xwn_scale"])) if xwn_bit == 1 else 0

    inst[0] &= 0xFFFF9FFF
    inst[0] |= (
        (xwn_scale << 9) |
        (xwn_bit << 14)
    )

def set_input_shape(blocks_config:dict, inst:np.ndarray, idx:int):
    shape = blocks_config["blocks"][idx]["input_shape"]
    channel_idx = blocks_config["blocks"][idx]["channel_idx"]
    if channel_idx is not None:
        shape[2] = (channel_idx[1] - channel_idx[0])

    inst[1] = (
        ((shape[2] & 0xFFF) << 20) |
        ((shape[0] & 0x3FF) << 10) |
        ((shape[1] & 0x3FF))
    )

def set_output_shape(blocks_config:dict, inst:np.ndarray, idx:int):
    shape = blocks_config["blocks"][idx]["output_shape"]

    inst[3] &= 0xF000FFFF
    inst[3] |= ((shape[2] & 0xFFF) << 16)

def set_kernel_shape(blocks_config:dict, inst:np.ndarray, idx:int):
    ker_h, ker_w, _, __= blocks_config["blocks"][idx]["kernel_shape"]
    is_diff = (ker_h != ker_w)

    inst[3] &= 0x0FFFFFF8
    inst[3] |= (
        (is_diff << 31) |
        ((ker_h & 0x7) << 28) |
        ((ker_w & 0x7) << 0)
    )

def set_stride(blocks_config:dict, inst:np.ndarray, idx:int):
    mode = blocks_config["blocks"][idx]["operation"]
    str_h, str_w = blocks_config["blocks"][idx]["stride_shape"]

    if mode == "TransposeConv":
        str_h, str_w = [1, 1]

    is_diff = (str_h != str_w)

    inst[2] &= 0x7FFFF0FF
    inst[2] |= (
        ((str_w & 0x3) << 8) |
        ((str_h & 0x3) << 10) |
        (is_diff << 31)
    )

def set_padding(blocks_config:dict, inst:np.ndarray, idx:int):
    pad_t, pad_l, pad_b, pad_r, pad_s = blocks_config["blocks"][idx]["padding_shape"]
    is_diff = ((pad_t + pad_b) != (pad_l + pad_r))
    # is_diff = 0

    inst[2] &= 0xFFFF8F88
    inst[3] &= 0xFFFF08FF

    inst[2] |= (
        (((pad_r) & 0x7) << 0) |
        ((pad_l & 0x7) << 4) | 
        ((pad_s & 0x7) << 12)
    )
    inst[3] |= (
        ((pad_b & 0x7) << 8) |
        ((pad_t & 0x7) << 12) |
        (is_diff << 15)
    )

def set_depth_separate_index(blocks_config:dict, inst:np.ndarray, idx:int):
    channel_index = blocks_config["blocks"][idx]["channel_idx"]
    channel_residual = blocks_config["blocks"][idx]["channel_residual"]

    start_depth = channel_index[0]
    end_depth   = channel_index[1]
    full_depth  = channel_index[2]

    if (end_depth - start_depth) == full_depth:
        return
    if (end_depth - start_depth) == (full_depth - channel_residual[1]):
        return
    if full_depth > 0xFFF:
        return

    inst[0] &= 0xFFFF7FFF
    inst[2] &= 0xF000FFFF

    inst[0] |= 0x00008000
    inst[2] |= ((full_depth & 0xFFF) << 16)

# batch normalization is always performed
def set_batch_noralization(blocks_config:dict, inst:np.ndarray, idx:int):
    return

def set_residual(blocks_config:dict, inst:np.ndarray, idx:int):
    residual_with = blocks_config["blocks"][idx]["residual_with"]
    if residual_with is not None:
        inst[0] &= 0xFFFFFEFF
        inst[0] |= (1 << 8)

def set_activate_function(blocks_config:dict, inst:np.ndarray, idx:int):
    inst[0] &= 0xFFFFFF0F
    relu = blocks_config["blocks"][idx]["activation"]

    if relu is not None:
        name = relu['name']
        if name == "Relu":
            inst[0] |= (1 << 4)
        elif name == "LeakyRelu":
            inst[0] |= (4 << 4)
        else:
            raise Exception("Unknown activate function name : {}".format(name))

def set_scaler(blocks_config:dict, inst:np.ndarray, idx:int):
    mode = blocks_config["blocks"][idx]["operation"]
    input_shape = blocks_config["blocks"][idx]["input_shape"]
    if mode == "Scaler":
        inst[2] &= 0xF000F0FF
        inst[3] &= 0xF000001F
        inst[7] &= 0x0

        inst[2] |= (
            (1 << 8) |
            (1 << 10) |
            (input_shape[1] << 16)
        )

        inst[3] |= (
            (input_shape[1] << 5) |
            (input_shape[2] << 16)
        )

        inst[6] |= (
            (1024 << 0) |
            (1024 << 16)
        )
