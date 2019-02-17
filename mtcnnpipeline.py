# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from prepare_data.genexamples import generate


# default directory for train data
dbdir = plib.Path(os.pardir).joinpath('data').absolute()


class DBWider:
    dir = dbdir.joinpath('WIDER_train', 'images')
    wider_face_train = dir.parent.joinpath('wider_face_train.txt')
    wider_face_train_bbx_gt = dir.parent.joinpath('wider_face_train_bbx_gt.txt')


class Models:
    epochs = (10, 30, 30)
    batch_size = (2048, 256, 16)
    dir = plib.Path(os.pardir).joinpath('mtcnn').absolute()
    path = (dir.joinpath('PNet', 'PNet'),
            dir.joinpath('RNet', 'RNet'),
            dir.joinpath('ONet', 'ONet'))


if __name__ == '__main__':
    seed = None

    # train PNet

    # ------------------------------------------------------------------------------------------------------------------
    # train O-Net (output net)

    # prepare examples
    threshold = (0.3, 0.1, 0.7)
    min_face_size = 20
    stride = 2

    h5file = dbdir.joinpath('onet', 'trainonet.h5')

    generate(h5file,
             DBWider(),
             Models(),
             mode='RNet',
             threshold=threshold,
             min_face_size=min_face_size,
             stride=stride,
             slide_window=False)


