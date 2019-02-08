# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from train_models.train import train
from train_models.mtcnn_model import P_Net


if __name__ == '__main__':

    base_dir = plib.Path(os.pardir).joinpath('data', '12').absolute()
    model_path = plib.Path(os.pardir).joinpath('data', '12', 'PNet').absolute()

    prefix = model_path
    number_of_epochs = 30
    display = 100
    lr = 0.001
    train('PNet', P_Net, prefix, number_of_epochs, base_dir, display=display, base_lr=lr)

