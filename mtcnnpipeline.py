# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from prepare_data import wider, lfw, tfrecords
from prepare_data.genexamples import generate
from train_models import pnet, rnet, onet
from train_models.train import train


# default directory to save train data
basedir = plib.Path(os.pardir).joinpath('dbase').absolute()


class DBNet:
    def __init__(self, basedir, dirname='PNet', label='pnet'):
        self.output = basedir.joinpath(dirname).absolute()
        self.h5file = self.output.joinpath(label + '.h5')
        self.tfrecord = self.output.joinpath(label)


class Models:
    epochs = (10, 30, 30)
    batch_size = (2048, 256, 16)
    dir = plib.Path(os.pardir).joinpath('mtcnn').absolute()
    path = (dir.joinpath('PNet', 'PNet'),
            dir.joinpath('RNet', 'RNet'),
            dir.joinpath('ONet', 'ONet'))


if __name__ == '__main__':
    seed = None

    # config for input wider and lfw data
    dbwider = wider.DBWider(basedir.joinpath('WIDER_train'))
    dblfw = lfw.DBLFW(basedir.joinpath('lfw'))

    # config for output data
    dbpnet = DBNet(basedir, dirname='PNet', label='dbpnet')
    dbrnet = DBNet(basedir, dirname='PNet', label='dbpnet')
    dbonet = DBNet(basedir, dirname='PNet', label='dbpnet')

    # ------------------------------------------------------------------------------------------------------------------
    # train PNet

    # prepare train data
    wider.prepare(dbwider, dbpnet, image_size=pnet.Config().image_size, seed=seed)
    lfw.prepare(dblfw, dbpnet, image_size=pnet.Config().image_size, seed=seed)

    # save tf record files
    labels = ('positive', 'part', 'negative', 'landmark')
    tfrecords.write_multi_tfrecords(dbpnet.h5file, dbpnet.tfrecord, labels, seed=None)

    # train
    prefix = plib.Path(os.pardir).joinpath('mtcnn', 'PNet', 'pnet').absolute()
    tffiles = tfrecords.getfilename(dbpnet.tfrecord, labels)
    train(pnet.Config(), tffiles, prefix)

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
