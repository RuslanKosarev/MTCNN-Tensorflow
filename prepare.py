# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from prepare_data.generate import generate12, GenerateData, gen_imglist
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

        self.postxt = self.posdir.parent.joinpath('pos_12.txt')
        self.negtxt = self.negdir.parent.joinpath('neg_12.txt')
        self.parttxt = self.partdir.parent.joinpath('part_12.txt')


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
    input = WiderData()
    output = WiderOutput()
    generate12(input, output)

    # the database contains the names of all the landmark training data
    input2 = LandmarkData()
    output2 = LandmarkOutput()
    GenerateData(input2, output2, 'Pnet', argument=True)

    data_dir = plib.Path(os.path.join(os.pardir, 'data/12')).absolute()
    gen_imglist(data_dir)

    dir = plib.Path(os.pardir).joinpath('data', '12').absolute()
    output_directory = dir.joinpath('PNet')
    tfrecords.generate(dir, output_directory, shuffling=False)
