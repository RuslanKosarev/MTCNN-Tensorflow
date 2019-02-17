# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from prepare_data.genexamples import generate
from prepare_data import dbwider


# default directory to save train data
basedir = plib.Path(os.pardir).joinpath('dbase').absolute()


class Models:
    epochs = (10, 30, 30)
    batch_size = (2048, 256, 16)
    dir = plib.Path(os.pardir).joinpath('mtcnn').absolute()
    path = (dir.joinpath('PNet', 'PNet'),
            dir.joinpath('RNet', 'RNet'),
            dir.joinpath('ONet', 'ONet'))


class PNetData:
    def __init__(self, basedir, path='PNet'):
        self.path = basedir.joinpath(path).absolute()
        self.h5file = self.path.joinpath('dbtrain.h5')


class RNetData:
    def __init__(self, basedir, path='RNet'):
        self.path = basedir.joinpath(path).absolute()
        self.h5file = self.path.joinpath('dbtrain.h5')


class ONetData:
    def __init__(self, basedir, path='ONet'):
        self.path = basedir.joinpath(path).absolute()
        self.h5file = self.path.joinpath('dbtrain.h5')


if __name__ == '__main__':
    seed = 0

    # ------------------------------------------------------------------------------------------------------------------
    # train PNet
    dbwider.prepare(dbwider.DBWider(basedir.joinpath('WIDER_train')),  # config for input wider data
                    PNetData(basedir),                                 # config for output data
                    seed=seed)

    # exit(0)
    # ------------------------------------------------------------------------------------------------------------------
    # train O-Net (output net)

    # # prepare examples
    # threshold = (0.3, 0.1, 0.7)
    # min_face_size = 20
    # stride = 2
    #
    # h5file = dbdir.joinpath('onet', 'trainonet.h5')
    #
    # generate(h5file,
    #          DBWider(),
    #          Models(),
    #          mode='RNet',
    #          threshold=threshold,
    #          min_face_size=min_face_size,
    #          stride=stride,
    #          slide_window=False)
    #
    #
