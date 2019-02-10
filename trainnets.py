# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from train_models import train


if __name__ == '__main__':

    base_dir = plib.Path(os.pardir).joinpath('data', '12').absolute()
    input = base_dir.joinpath('PNet')
    prefix = input.joinpath('PNet')

    number_of_epochs = 30
    display = 100
    lr = 0.001

    train.train_pnet(input, prefix, number_of_epochs, base_dir, display=display, base_lr=lr)

