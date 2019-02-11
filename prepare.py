# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
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

        # output directory to wider database
        if outdir is None:
            outdir = plib.Path(os.pardir).joinpath('data', '12')
        self.outdir = plib.Path(outdir).absolute()

        self.positive = self.outdir.joinpath('positive')
        self.negative = self.outdir.joinpath('negative')
        self.part = self.outdir.joinpath('part')

        for name in (self.positive, self.negative, self.part):
            if not name.exists():
                name.mkdir(parents=True)

        self.h5out = self.outdir.joinpath('wider.h5')


class LandmarkData:
    def __init__(self, dir=None):
        if dir is not None:
            self.dir = plib.Path(dir).absolute()
        else:
            self.dir = plib.Path(os.pardir).joinpath('data/FacialLandmarks').absolute()
        self.annotations = self.dir.joinpath('trainImageList.txt')


class LandmarkOutput:
    def __init__(self):
        self.dir = plib.Path(os.pardir).joinpath('data/12').absolute()
        self.files = self.dir.joinpath('train_PNet_landmark_aug')
        self.annotations = self.dir.joinpath('landmark_12_aug.txt')


if __name__ == '__main__':
    # seed to generate random numbers
    seed = None

    dbase = WiderDBase()
    from datetime import datetime
    start = datetime.now()
    prepare.widerdbase(dbase, seed=seed)
    print(datetime.now() - start)
    exit(0)

    # the database contains the names of all the landmark training data
    input2 = LandmarkData()
    output2 = LandmarkOutput()
    prepare.lfwdbase(input2, output2, 'Pnet', argument=True, seed=seed)

    data_dir = plib.Path(os.path.join(os.pardir, 'data/12')).absolute()
    #mergedbase(data_dir)
    exit(0)

    dir = plib.Path(os.pardir).joinpath('data', '12').absolute()
    output_directory = dir.joinpath('PNet')
    tfrecords.generate(dir, output_directory, shuffling=True, seed=seed)
