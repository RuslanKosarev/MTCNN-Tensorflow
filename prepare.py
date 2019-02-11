# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from prepare_data.generate import wider, GenerateData, mergedbase
from prepare_data import tfrecords


class WiderData:
    def __init__(self, dir=None):
        if dir is not None:
            self.images = plib.Path(dir).absolute()
        else:
            self.images = plib.Path(os.pardir).joinpath('data/WIDER_train/images').absolute()

        self.annotation = self.images.parent.joinpath('wider_face_train.txt')


class WiderOutput:
    def __init__(self, dir=None):
        if dir is not None:
            self.dir = plib.Path(dir).absolute()
        else:
            self.dir = plib.Path(os.pardir).joinpath('data').absolute()

        self.outdir = plib.Path(self.dir.joinpath('12'))
        self.posdir = self.outdir.joinpath('positive')
        self.negdir = self.outdir.joinpath('negative')
        self.partdir = self.outdir.joinpath('part')
        self.landmark = self.outdir.joinpath('train_PNet_landmark_aug')

        for dir in (self.posdir, self.negdir, self.partdir, self.landmark):
            if not dir.exists():
                dir.mkdir(parents=True)

        self.postxt = self.posdir.parent.joinpath('positive.txt')
        self.negtxt = self.negdir.parent.joinpath('negative.txt')
        self.parttxt = self.partdir.parent.joinpath('part.txt')

        self.h5outfile = self.outdir.joinpath('wider.h5')


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

    input = WiderData()
    output = WiderOutput()
    from datetime import datetime
    start = datetime.now()
    wider(input, output, seed=seed)
    print(datetime.now() - start)
    exit(0)

    # the database contains the names of all the landmark training data
    input2 = LandmarkData()
    output2 = LandmarkOutput()
#    GenerateData(input2, output2, 'Pnet', argument=True, seed=seed)

    data_dir = plib.Path(os.path.join(os.pardir, 'data/12')).absolute()
    mergedbase(data_dir)
    exit(0)

    dir = plib.Path(os.pardir).joinpath('data', '12').absolute()
    output_directory = dir.joinpath('PNet')
    tfrecords.generate(dir, output_directory, shuffling=True, seed=seed)
