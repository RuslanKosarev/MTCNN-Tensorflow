# coding:utf-8

import numpy as np
import h5py
from easydict import EasyDict as edict

pnet_dtype = np.dtype([('path', h5py.special_dtype(vlen=str)),
                       ('flag', np.int32),
                       ('1', np.float),
                       ('2', np.float),
                       ('3', np.float),
                       ('4', np.float)])

config = edict()
config.BATCH_SIZE = 384
config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [6,14,20]
