import numpy as np

def align_odd_width(block_config):
    _, in_w, _ = block_config["input_shape"]
    ker_h, ker_w, _, _ = block_config["kernel_shape"]
    str_h, str_w = block_config["stride_shape"]
    p_t, p_l, p_b, p_r, _ = block_config["padding_shape"]

    is_odd = (in_w % 2 == 1)
    is_ker_1x1 = (ker_h == 1 and ker_w == 1)
    is_str_2x2 = (str_h == 2 and str_w == 2)
    is_no_pad = ((p_t == 0) and (p_l == 0) and (p_b == 0) and (p_r == 0))

    if is_odd and is_ker_1x1 and is_str_2x2 and is_no_pad:
        block_config["padding_shape"][2] += 1 # add bottom padding
        block_config["padding_shape"][3] += 1 # add right padding

    return block_config

def add_dummy_padding(block_config):
    in_h, in_w, _       = block_config["input_shape"]
    out_h, out_w, _     = block_config["output_shape"]
    ker_h, ker_w, _, __ = block_config["kernel_shape"]
    str_h, str_w        = block_config["stride_shape"]
    p_t, p_l, p_b, p_r, p_s = block_config["padding_shape"]

    op = block_config["operation"]
    if op == "Conv":
        if out_w < 7:
            for i in range(8):
                if (1 + (in_w - 1 + p_l + p_r + i - (ker_w - 1)) / str_w >= 7):
                    p_s = i
                    break
            p_r += p_s
    else:
        if op == "MaxPool": # Warning: Only for synergy ai
            if ker_w+1 > in_w:
                add_pad = max((ker_w + 1 - in_w), 0)
                p_r += add_pad
                p_s += ((in_w + p_r) - ker_w)

    block_config["padding_shape"] = [p_t, p_l, p_b, p_r, p_s]
    return block_config

    # def add_dummy_padding(block_config):
    #     in_h, in_w, _ = block_config["input_shape"]
    #     ker_h, ker_w, _, __ = block_config["kernel_shape"]
    # 
    #     p_t, p_l, p_b, p_r, p_s = block_config["padding_shape"]
    # 
    #     op = block_config["operation"]
    #     if op == "Conv":
    #         if in_w < 7:
    #             add_pad = (7 - in_w)
    #             p_r += add_pad
    #             p_s += add_pad
    #     else:
    #         if op == "MaxPool": # Warning: Only for synergy ai
    #             if ker_w+1 > in_w:
    #                 add_pad = max((ker_w + 1 - in_w), 0)
    #                 p_r += add_pad
    #                 p_s += ((in_w + p_r) - ker_w)
    # 
    #     block_config["padding_shape"] = [p_t, p_l, p_b, p_r, p_s]
    #     return block_config
