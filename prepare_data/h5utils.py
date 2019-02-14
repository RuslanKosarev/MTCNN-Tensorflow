# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import h5py
from datetime import datetime
import numpy as np
import imageio
import pathlib as plib
import os


def write_image(hf, name, image, mode='a', check_name=True):
    with h5py.File(str(hf), mode) as hf:

        if name in hf and check_name is True:
            raise IOError('data set {} has already existed'.format(name))

        if not name in hf:
            dset = hf.create_dataset(name=name,
                                     data=image,
                                     dtype='uint8',
                                     compression='gzip',
                                     compression_opts=9)
        else:
            hf[name][...] = image


def write(filename, name, data, mode='a'):
    with h5py.File(str(filename), mode=mode) as hf:
        if name in hf:
            del hf[name]

        hf.create_dataset(name,
                          data=data,
                          shape=data.shape,
                          maxshape=(None,),
                          compression='gzip',
                          dtype=data.dtype)

    fullkey = os.path.join(str(filename), name)
    print('dataset \'{}\' has been written to the file {} (length {})'.format(name, fullkey, len(data)))


def read(filename, name):
    with h5py.File(str(filename), mode='r') as hf:
        return hf[name][...]
