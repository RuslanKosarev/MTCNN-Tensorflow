# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from datetime import datetime
from prepare_data import tfrecords


if __name__ == '__main__':
    seed = 0
    outdir = plib.Path(os.pardir).joinpath('data', '12').absolute()
    h5file = 'dbtrain.h5'

    start = datetime.now()
    h5file = outdir.joinpath(h5file)
    tfrecord = outdir.joinpath('dbtrain.tfrecord')
    tfrecords.pnet_tfrecord(h5file, tfrecord, seed=seed)
    print(datetime.now() - start)
