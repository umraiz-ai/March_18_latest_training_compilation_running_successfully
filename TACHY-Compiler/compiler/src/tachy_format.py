#coding:utf-8


import numpy as np
# import _pickle as pickle
import pickle


class DictionaryTACHY(dict):

    IS_INIT = False

    def __init__(self, *args, **kwargs):
        super(DictionaryTACHY, self).__init__(*args, **kwargs)
        self.itemlist = list(super(DictionaryTACHY, self).keys())
        self.IS_INIT = True

    def __setitem__(self, key, value):
        if self.IS_INIT:
            self.itemlist.append(key)
        else:
            self.itemlist = []
            self.IS_INIT = True
    
        super(DictionaryTACHY, self).__setitem__(key, value)

    def __iter__(self):
        return iter(self.itemlist)

    def keys(self):
        return self.itemlist

    def values(self):
        return [self[key] for key in self]

    def itervalues(self):
        return (self[key] for key in self)


def save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

'''
Note that memory allocation(__new__) and initialization(__init__) are not performed during load execution.
'''
def load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data


tdict = DictionaryTACHY
tsave = save
tload = load

if __name__ == '__main__':
    tachy_dict = tdict(a=1, b=2)
    tsave('test.tachy', tachy_dict)
    print(tload('test.tachy'))

