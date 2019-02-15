# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from train_models import train
import config


if __name__ == '__main__':
    base_dir = plib.Path(os.pardir).joinpath('data', 'rnet').absolute()
    tfrecord = plib.Path(os.pardir).joinpath('data', 'rnet', 'rnet').absolute()

    prefix = plib.Path(os.pardir).joinpath('mtcnn', 'RNet', 'RNet').absolute()

    netconfig = config.RNetConfig()
    keys = ('positive', 'negative', 'part', 'landmark')

    tfrecords = []
    for key in keys:
        tfrecords.append(tfrecord.with_name(tfrecord.name + key).with_suffix('.tfrecord'))

    train.train(netconfig, tfrecords, prefix)
