# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from train_models import train


if __name__ == '__main__':
    base_dir = plib.Path(os.pardir).joinpath('data', 'rnet').absolute()
    prefix = plib.Path(os.pardir).joinpath('mtcnn', 'RNet', 'RNet').absolute()

    number_of_epochs = 30
    lr = 0.001

    tfrecord = base_dir.joinpath('dbtrain.tfrecord')
    train.train(tfrecord, prefix, number_of_epochs, lr=lr)

