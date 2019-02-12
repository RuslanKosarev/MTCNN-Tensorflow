# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from datetime import datetime
from prepare_data import prepare
from prepare_data import tfrecords


class WiderDBase:
    def __init__(self, inpdir=None, outdir=None, h5file=None):
        # input directory to wider database
        if inpdir is None:
            inpdir = plib.Path(os.pardir).joinpath('data', 'WIDER_train')
        inpdir = plib.Path(inpdir).absolute()

        self.images = inpdir.joinpath('images')
        self.annotation = inpdir.joinpath('wider_face_train.txt')

        # output directory for wider database
        if outdir is None:
            self.outdir = plib.Path(os.pardir).joinpath('data', '12')
        else:
            self.outdir = plib.Path(outdir).absolute()

        self.positive = self.outdir.joinpath('positive')
        self.negative = self.outdir.joinpath('negative')
        self.part = self.outdir.joinpath('part')

        if h5file is None:
            self.h5out = self.outdir.joinpath('dbwider.h5')
        else:
            self.h5out = self.outdir.joinpath(h5file)

        for name in (self.positive, self.negative, self.part):
            if not name.exists():
                name.mkdir(parents=True)


class LFWDBase:
    def __init__(self, inpdir=None, outdir=None, h5file=None):
        # input directory to LFW database
        if inpdir is None:
            inpdir = plib.Path(os.pardir).joinpath('data', 'lfw').absolute()
        self.inpdir = plib.Path(inpdir).absolute()
        self.annotations = self.inpdir.joinpath('trainImageList.txt')

        # output directory for LFW database
        if outdir is None:
            self.outdir = plib.Path(os.pardir).joinpath('data', '12').absolute()
        else:
            self.outdir = plib.Path(outdir).absolute()

        if h5file is None:
            self.h5file = self.outdir.joinpath('dblfw.h5')
        else:
            self.h5file = self.outdir.joinpath(h5file)

        self.keys = ('landmarks',)
        for key in self.keys:
            if not self.outdir.joinpath(key).exists():
                self.outdir.joinpath(key).mkdir(parents=True)


if __name__ == '__main__':

    seed = None
    outdir = plib.Path(os.pardir).joinpath('data', '12')
    h5file = 'dbtrain.h5'

    # prepare wider database
    wider = WiderDBase(outdir=outdir, h5file=h5file)
    start = datetime.now()
    prepare.widerdbase(wider, seed=seed)
    print(datetime.now() - start)

    # prepare lfw database
    lfw = LFWDBase(outdir=outdir, h5file=h5file)
    start = datetime.now()
    prepare.lfwdbase(lfw, seed=seed)
    print(datetime.now() - start)
