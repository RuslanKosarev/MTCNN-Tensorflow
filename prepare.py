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
        self.h5out = self.outdir.joinpath('wider.h5')

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
        self.h5txt = self.outdir.parent.joinpath('landmark.txt')
        self.h5out = self.outdir.parent.joinpath('lfw.h5')

        for name in (self.outdir,):
            if not name.exists():
                name.mkdir(parents=True)


if __name__ == '__main__':

    # dbase = WiderDBase()
    # start = datetime.now()
    # prepare.widerdbase(dbase, seed=seed)
    # print(datetime.now() - start)

    # the database contains the names of all the landmark training data
    dbase = LFWDBase()
    start = datetime.now()
    prepare.lfwdbase(dbase)
    print(datetime.now() - start)

    # data_dir = plib.Path(os.path.join(os.pardir, 'data/12')).absolute()
    # #mergedbase(data_dir)
    # exit(0)
    #
    # dir = plib.Path(os.pardir).joinpath('data', '12').absolute()
    # output_directory = dir.joinpath('PNet')
    # tfrecords.generate(dir, output_directory, shuffling=True, seed=seed)
