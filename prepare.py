# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from prepare_data.generate import generate12


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

        for dir in (self.posdir, self.negdir, self.partdir):
            if not dir.exists():
                dir.mkdir(parents=True)

        self.postxt = self.posdir.parent.joinpath('pos_12.txt')
        self.negtxt = self.negdir.parent.joinpath('neg_12.txt')
        self.parttxt = self.partdir.parent.joinpath('part_12.txt')


if __name__ == '__main__':
    input = WiderData()
    output = WiderOutput()

    generate12(input, output)
