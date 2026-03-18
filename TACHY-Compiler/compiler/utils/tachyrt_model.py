import base64
import datetime
import commentjson
import numpy as np
from easydict import EasyDict

class tachyrt_model:
    def __init__(self):
        self.partition = {
            "header": 0,
            "instruction": 1,
            "descriptor": 2,
            "parameter": 3
        }
        self.__version__ = 2
        print("init tachyrt_model v{}".format(float(self.__version__)))

    def load_model(self, file):
        data = np.fromfile(file, np.uint8)

        offset = 0
        while offset < data.nbytes:
            partition, length = data[offset:offset+8].view(np.uint32)[:2]
            offset += 8

            self._parse(partition, data[offset:offset + length])
            offset += length

        return self

    def build_model(self, inst: np.array, desc: dict, param: np.array):
        assert(type(inst) == np.ndarray)
        assert(type(desc) == dict), print(type(desc))
        assert(type(param) == np.ndarray)

        _dict = {}
        self._header = self._set_header(_dict)
        self._inst   = self._set_inst(_dict, inst)
        self._desc   = self._set_desc(_dict, desc)
        self._param  = self._set_param(_dict, param)

        data_header = np.frombuffer(commentjson.dumps(self._header).encode(), np.uint8)
        data_inst = self._inst
        data_desc = np.frombuffer(commentjson.dumps(self._desc).encode(), np.uint8)
        data_param = self._param
        size_header = len(data_header)
        size_inst = len(data_inst)
        size_desc = len(data_desc)
        size_param = len(data_param)

        data = np.concatenate([
            np.array([self.partition["header"]], np.uint32).view(np.uint8),
            np.array([size_header], np.uint32).view(np.uint8),
            data_header,

            np.array([self.partition["instruction"]], np.uint32).view(np.uint8),
            np.array([size_inst], np.uint32).view(np.uint8),
            data_inst,

            np.array([self.partition["descriptor"]], np.uint32).view(np.uint8),
            np.array([size_desc], np.uint32).view(np.uint8),
            data_desc,

            np.array([self.partition["parameter"]], np.uint32).view(np.uint8),
            np.array([size_param], np.uint32).view(np.uint8),
            data_param,
        ])

        return data

    def get_info(self):
        '''
        1. header
        '''
        print(self._header)

    def _set_header(self, _dict):
        _dict["header"] = {}
        _dict["header"]["date"] = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        _dict["header"]["bs_ver"] = 2
        _dict["header"]["rt_ver"] = 2

        return _dict["header"]

    def _set_inst(self, _dict, inst):
        '''
        1. instruction partition
        2. instruction partition size
        3. instruction data
        '''
        inst = inst.view(np.uint8).reshape(-1)
        assert((inst.size % 32) == 0), "Instruction must be ended without exit layer"

        return inst

    def _set_desc(self, _dict, desc):
        '''
        1. descriptor partition
        2. descriptor partition size
        2. descriptor n_layer
        3. descriptor data
        '''
        return desc

    def _set_param(self, _dict, param):
        '''
        1. parameter partition
        2. parameter partition size
        3. parameter data
        '''
        param = param.view(np.uint8).reshape(-1)
        return param

    def _get_rt_ver(self):
        return self.__version__

    def _parse(self, partition, data):
        if self.partition["header"] == partition:
            self._parse_header(data)
        elif self.partition["instruction"] == partition:
            self._parse_instruction(data)
        elif self.partition["descriptor"] == partition:
            self._parse_descriptor(data)
        elif self.partition["parameter"] == partition:
            self._parse_parameter(data)

    def _parse_header(self, data):
        '''
        1. header data(name) -> 30 byte
        2. header data(date) -> 14 byte
        2. header data(bs_ver) -> 4 byte
        3. header data(rt_ver) -> 4 byte
        '''
        self._header = commentjson.loads(data.tobytes().decode())

    def _parse_instruction(self, data):
        '''
        1. instruction data
        '''
        self._instruction = data[:]

        return True

    def _parse_descriptor(self, data):
        '''
        1. descriptor layer len
        2. descriptor data
        '''
        self._descriptor = EasyDict(commentjson.loads(data.tobytes().decode()))

    def _parse_parameter(self, data):
        '''
        1. parameter data
        '''
        self._parameter = data[:]

        return True

    def _extract_desc_from_inst(self, inst):
        pass

    def get_n_layer_inst(self, n_layer):
        offset = n_layer * 32
        ret = self._instruction[offset:offset+32].view(np.uint32)
        return ret

    def get_n_layer_param(self, n_layer):
        offset = n_layer * 8
        offset_next = (n_layer+1) * 8
        base = self._instruction.view(np.uint32)[5].byteswap()
        start = self._instruction.view(np.uint32)[offset+5].byteswap()
        end = -1 if n_layer == (self.n_layer-1) else self._instruction.view(np.uint32)[offset_next+5].byteswap()

        offset = start - base
        size = end - start

        ret = self._parameter[offset:offset+size]
        return ret

    def get_n_layer_input_size(self, n_layer):
        size = int(np.prod(self._descriptor[n_layer].input_shape)) * 2
        return size

    def get_n_layer_output_size(self, n_layer):
        size = int(np.prod(self._descriptor[n_layer].output_shape)) * 2
        return size

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def instruction(self):
        return self._instruction

    @property
    def parameter(self):
        return self._parameter

    @property
    def n_layer(self):
        return self._descriptor["n_block"]

    @property
    def descriptor(self):
        return self._descriptor

if __name__ == '__main__':
    mod = tachyrt_model()
    mod.load_model('/mnt/TACHY-Station/object_detection_yolov40/TACHY-H200/CharKR82-20221013_0-YOLOv4_D/model_80x192x3_inv-f.tachyrt')

    import pprint
    pprint.pprint(mod.descriptor.keys())

    mod.descriptor['use_scaler'] = [ True ]
    mod.descriptor['input_format'] = [ "image" ]
    mod.descriptor['mode_gap'] = "serial"
    mod.descriptor['multiple'] = 8
    mod.descriptor['pad_order'] = [ "down_right" ]
    mod.descriptor['logit_order'] = [ 1, 0 ]
    mod.descriptor['use_input_inv'] = False
    mod.descriptor['use_opt_precision'] = False

    mod.descriptor['blocks'] = []
    mod.descriptor['input_block'] = []
    mod.descriptor['input_shape'] = []
    mod.descriptor['output_block'] = []
    mod.descriptor['output_shape'] = []
    for i in range(mod.descriptor['n_block']):
        if mod.descriptor[str(i)]['is_start']:
            mod.descriptor['input_block'].append(i)
            mod.descriptor['input_shape'].append(mod.descriptor[str(i)]['input_shape'])

        if mod.descriptor[str(i)]['is_logit']:
            mod.descriptor['output_block'].append(i)
            mod.descriptor['output_shape'].append(mod.descriptor[str(i)]['output_shape'] + mod.descriptor[str(i)]['kernel_shape'][3:])

        mod.descriptor['blocks'].append(mod.descriptor[str(i)])
        del mod.descriptor[str(i)]


    print(mod.descriptor['n_block'])
    assert('n_block' in mod.descriptor)

    print(mod.descriptor['use_scaler'])
    assert('use_scaler' in mod.descriptor)

    print(mod.descriptor['input_format'])
    assert('input_format' in mod.descriptor)

    print(mod.descriptor['input_block'])
    assert('input_block' in mod.descriptor)

    print(mod.descriptor['input_shape'])
    assert('input_shape' in mod.descriptor)

    print(mod.descriptor['output_block'])
    assert('output_block' in mod.descriptor)

    print(mod.descriptor['output_shape'])
    assert('output_shape' in mod.descriptor)

    print(mod.descriptor['mode_gap'])
    assert('mode_gap' in mod.descriptor)

    assert('multiple' in mod.descriptor)
    assert('pad_order' in mod.descriptor)
    assert('logit_order' in mod.descriptor)
    assert('use_input_inv' in mod.descriptor)
    assert('use_opt_precision' in mod.descriptor)

    import pprint
    pprint.pprint(mod.descriptor)

    data = mod.build_model(mod.instruction, dict(mod.descriptor), mod.parameter)
    data.tofile('/mnt/TACHY-Station/object_detection_yolov40/TACHY-H200/CharKR82-20221013_0-YOLOv4_D/model_80x192x3_inv-f.tachyrt')
