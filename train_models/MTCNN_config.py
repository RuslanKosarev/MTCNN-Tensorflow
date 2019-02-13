# coding:utf-8

import numpy as np
import h5py
from easydict import EasyDict as edict

wider_dtype = np.dtype([('path', h5py.special_dtype(vlen=str)),
                        ('label', np.int8),
                        ('1', np.float),
                        ('2', np.float),
                        ('3', np.float),
                        ('4', np.float)])

lfw_dtype = np.dtype([('path', h5py.special_dtype(vlen=str)),
                      ('label', np.int8),
                      ('1', np.float),
                      ('2', np.float),
                      ('3', np.float),
                      ('4', np.float),
                      ('5', np.float),
                      ('6', np.float),
                      ('7', np.float),
                      ('8', np.float),
                      ('9', np.float),
                      ('10', np.float)])

config = edict()
config.BATCH_SIZE = 384
config.tfrecord_size = 100000
config.pnet_ratio = [1, 3, 1, 1]


config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [6,14,20]
