#coding:utf-8

import os, sys
sys.path.append('./src')

import json

from tachy_format import *
from tachy_model import TachyModel


if __name__ == '__main__':
    src_file = 'comps.tachy' 
    key = 'tachy_model'
    tachy_dict = tload(src_file)
    tachy_model = tachy_dict[key]
    for n_idx in tachy_model.graph.nodes:
        # print(tachy_model.graph.nodes[n_idx])
        print(tachy_model.graph.nodes[n_idx].keys())
        # print(tachy_model.graph.nodes[n_idx]['params_sign'])
        print(tachy_model.graph.nodes[n_idx]['params_scale'])
        # print(tachy_model.graph.nodes[n_idx]['params_magnitude'])

    # for k in tachy_dict['TACHY'][key]:
    #     print(k, tachy_dict['TACHY'][key][k].shape, tachy_dict['TACHY'][key][k].dtype)

