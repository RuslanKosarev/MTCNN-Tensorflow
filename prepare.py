# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from datetime import datetime
from prepare_data import prepare
from prepare_data import tfrecords


class WiderDBase:
    def __init__(self, inpdir=None, outdir=None):
        # input directory to wider database
        if inpdir is None:
            inpdir = plib.Path(os.pardir).joinpath('data', 'WIDER_train')
        inpdir = plib.Path(inpdir).absolute()

        self.images = inpdir.joinpath('images')
        self.annotation = inpdir.joinpath('wider_face_train.txt')

        # output directory for wider database
        if outdir is None:
            outdir = plib.Path(os.pardir).joinpath('data', '12')
        self.outdir = plib.Path(outdir).absolute()

        self.positive = self.outdir.joinpath('positive')
        self.negative = self.outdir.joinpath('negative')
        self.part = self.outdir.joinpath('part')
        self.h5out = self.outdir.joinpath('dbwider.h5')

        for name in (self.positive, self.negative, self.part):
            if not name.exists():
                name.mkdir(parents=True)


class LFWDBase:
    def __init__(self, inpdir=None, outdir=None):
        # input directory to LFW database
        if inpdir is None:
            inpdir = plib.Path(os.pardir).joinpath('data', 'lfw').absolute()
        self.inpdir = plib.Path(inpdir).absolute()
        self.annotations = self.inpdir.joinpath('trainImageList.txt')

        # output directory for LFW database
        if outdir is None:
            outdir = plib.Path(os.pardir).joinpath('data', '12').absolute()
        outdir = plib.Path(outdir).absolute()

        self.outdir = plib.Path(outdir).joinpath('lfw').absolute()
        self.h5out = self.outdir.parent.joinpath('dblfw.h5')

        for name in (self.outdir,):
            if not name.exists():
                name.mkdir(parents=True)


if __name__ == '__main__':
    seed = None

    # prepare wider database
    wider = WiderDBase()
    start = datetime.now()
    prepare.widerdbase(wider, seed=seed)
    print(datetime.now() - start)

    # prepare lfw database
    lfw = LFWDBase()
    start = datetime.now()
    prepare.lfwdbase(lfw, seed=seed)
    print(datetime.now() - start)

    # merge databases
    train = wider.outdir.joinpath('dbtrain.h5')
    prepare.merge(train, wider=wider, lfw=lfw)
