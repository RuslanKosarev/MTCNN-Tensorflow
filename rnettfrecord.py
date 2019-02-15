# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from datetime import datetime
from prepare_data import tfrecords
import mtcnn_config


if __name__ == '__main__':
    seed = None

    h5file = plib.Path(os.pardir).joinpath('data', 'rnet', 'dbtrain.h5').absolute()
    prefix = plib.Path(os.pardir).joinpath('data', 'rnet', 'rnet').absolute()

    keys = ('positive', 'negative', 'part', 'landmark')

    for key in keys:
        tfrecord = prefix.with_name(prefix.name + key).with_suffix('.tfrecord')
        tfrecords.tfrecords(h5file, tfrecord, keys=key, sizes=None, seed=seed)

