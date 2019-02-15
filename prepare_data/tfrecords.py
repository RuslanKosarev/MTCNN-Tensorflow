# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import sys
import numpy as np
import tensorflow as tf
from prepare_data.tfrecord_utils import process_image_without_coder, _convert_to_example_simple
from prepare_data import h5utils
from train_models.MTCNN_config import config


def add_to_tfrecord(writer, filename, data):
    image_data, height, width = process_image_without_coder(filename)
    example = _convert_to_example_simple(data, image_data)
    writer.write(example.SerializeToString())


def data2sample(inpdata):

    outdata = []
    rect = ('xmin', 'ymin', 'xmax', 'ymax')
    landmarks = ('xlefteye', 'ylefteye',
                 'xrighteye', 'yrighteye',
                 'xnose', 'ynose',
                 'xleftmouth', 'yleftmouth',
                 'xrightmouth', 'yrightmouth')

    for values in inpdata:
        sample = dict()
        sample['filename'] = values[0]
        sample['label'] = values[1]
        sample['bbox'] = {x: 0 for x in rect + landmarks}

        values = list(values)[2:]

        if len(values) == 4:
            for key, value in zip(rect, values):
                sample['bbox'][key] = value
        else:
            for key, value in zip(landmarks, values):
                sample['bbox'][key] = value
        outdata.append(sample)

    return outdata


def tfrecords(h5file, tfrecord, keys=None, sizes=None, seed=None):
    """

    :param h5file:
    :param tfrecord:
    :param keys:
    :param sizes:
    :param seed:
    :return:
    """
    np.random.seed(seed=seed)

    # tf record file name
    if tfrecord.exists():
        os.remove(str(tfrecord))

    # get data from the h5 file
    if keys is None:
        keys = ('positive', 'negative', 'part', 'landmark')
    if not isinstance(keys, (list, tuple)):
        keys = (keys,)

    if sizes is None:
        sizes = [None]*len(keys)

    tfdata = []

    for key, size in zip(keys, sizes):
        data = h5utils.read(h5file, key)

        if size is None:
            size = len(data)
        if size < len(data):
            data = np.random.choice(data, size=size)

        tfdata += data2sample(data)

    np.random.shuffle(tfdata)

    with tf.python_io.TFRecordWriter(str(tfrecord)) as writer:
        for i, sample in enumerate(tfdata):
            filename = h5file.parent.joinpath(sample['filename'])
            add_to_tfrecord(writer, filename, sample)

            if (i+1) % 100 == 0:
                print('\r{}/{} samples have been added to tfrecord file.'.format(i+1, len(tfdata)), end='')

    print('\rtfrecord file {} has been written, batch size is {}.'.format(tfrecord, len(tfdata)))


