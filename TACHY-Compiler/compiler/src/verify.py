def verify_parameter_size(config):
    if config["operation"] != "MaxPool" and config["operation"] != "Concat":
        assert(config["param_size"] > 0), "Block_{} parameter error".format(config["bi"])

    return True

def verify_sram_size(config, sram_size=4096):
    ''' TODO: add width padding to input width'''
    bi = config["bi"]
    stt, end, total = config["channel_idx"]
    in_h, in_w, _ = config["input_shape"]
    in_d = end - stt
    assert((in_w * in_d) <= sram_size), "Block_{} input size({}x{}x{}) is bigger than sram size({})".format(bi, in_h, in_w, in_d, sram_size)

    return True
