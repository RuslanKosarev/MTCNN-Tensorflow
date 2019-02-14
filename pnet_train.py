# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from train_models import train


if __name__ == '__main__':
    base_dir = plib.Path(os.pardir).joinpath('data', '12').absolute()
    prefix = base_dir.joinpath('PNet', 'PNet')

    number_of_steps = 10
    display = 100
    lr = 0.001

    tfrecord = base_dir.joinpath('dbtrain.tfrecord')
    train.train_pnet(tfrecord, prefix, number_of_steps, display=display, lr=lr)

