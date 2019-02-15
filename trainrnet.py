# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from train_models import train
import config


if __name__ == '__main__':
    prefix = plib.Path(os.pardir).joinpath('mtcnn', 'RNet', 'RNet').absolute()

    netconfig = config.RNetConfig()

    keys = ('positive', 'negative', 'part', 'landmark')
    tfrecord = plib.Path(os.pardir).joinpath('data', 'rnet', 'rnet').absolute()
    tfrecords = []

    for key in keys:
        tfrecords.append(tfrecord.with_name(tfrecord.name + key).with_suffix('.tfrecord'))

    train.train(netconfig, tfrecords, prefix)
