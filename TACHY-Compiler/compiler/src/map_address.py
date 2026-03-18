def set_input_addr(map, blocks_config, idx):
    input_from = blocks_config['blocks'][idx]['input_from']
    offset_depth_sep = blocks_config['blocks'][idx]['channel_idx'][0] * blocks_config['blocks'][idx]['input_shape'][1] * 2

    if input_from == -1:
        map[idx]['input_addr'] = 0x1100_0000
    else:
        map[idx]['input_addr'] = map[input_from]['output_addr'] + offset_depth_sep

def set_param_addr(map, blocks_config, idx):
    map[idx]['param_addr'] = 0x1d00_0000 + blocks_config["blocks"][idx]["param_offset"]

def set_resi_addr(map, blocks_config, idx):
    residual_index = blocks_config['blocks'][idx]['residual_with']
    if residual_index is not None:
        map[idx]['residual_addr'] = map[residual_index]['output_addr']

def set_output_addr(map, blocks_config, idx):
    if idx == 0:
        map[idx]['output_addr'] = 0x2000_0000
        return

    is_logit = blocks_config["blocks"][idx]["is_logit"]
    if is_logit:
        if map['first_logit']:
            map[idx]['output_addr'] = 0x2500_0000
            map['first_logit'] = False
            return

        h, w, d, _ = blocks_config["output_shape"][blocks_config["output_block"].index(idx) - 1]
        size = h * w * d * 2

        map[idx]['output_addr'] = 0x2500_0000 + map['logit_offset'] + size
        map['logit_offset'] += size
    else:
        h, w, d = blocks_config["blocks"][idx-1]["output_shape"]
        h = h if h % 8 == 0 else h + (8 - (h % 8))
        size = h * w * d * 2

        if size % 0x1000 != 0:
            size = size + (0x1000 - (size % 0x1000))

        map[idx]['output_addr'] = 0x2000_0000 + map['out_offset'] + size
        map['out_offset'] += size
