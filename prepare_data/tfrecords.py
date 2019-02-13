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


def get_dataset(inpdata):

    outdata = []
    rect = ('xmin', 'ymin', 'xmax', 'ymax')
    landmarks = ('xlefteye', 'ylefteye',
                 'xrighteye', 'yrighteye',
                 'xnose', 'ynose',
                 'xleftmouth', 'yleftmouth',
                 'xrightmouth', 'yrightmouth')

    for values in inpdata:
        bbox = {x: 0 for x in rect + landmarks}

        if len(values) == 6:
            bbox['xmin'] = values[2]
            bbox['ymin'] = values[3]
            bbox['xmax'] = values[4]
            bbox['ymax'] = values[5]
        else:
            bbox['xlefteye'] = values[2]
            bbox['ylefteye'] = values[3]
            bbox['xrighteye'] = values[4]
            bbox['yrighteye'] = values[5]
            bbox['xnose'] = values[6]
            bbox['ynose'] = values[7]
            bbox['xleftmouth'] = values[8]
            bbox['yleftmouth'] = values[9]
            bbox['xrightmouth'] = values[10]
            bbox['yrightmouth'] = values[11]

        sample = dict()
        sample['filename'] = values[0]
        sample['label'] = values[1]
        sample['bbox'] = bbox
        outdata.append(sample)

    return outdata


def pnet_tfrecord(h5file, tfrecord, seed=None):
    """

    :param tfrecord:
    :param h5file:
    :param seed:
    :return:
    """
    np.random.seed(seed=seed)

    # tf record file name
    if tfrecord.exists():
        os.remove(str(tfrecord))

    # get data from the h5 file
    keys = ('positive', 'negative', 'part', 'landmark')
    # ratios = np.array([1, 3, 1, 1])
    sizes = np.array([1, 3, 1, np.NaN])*250000
    tfdata = []

    for key, size in zip(keys, sizes):
        data = h5utils.read(h5file, key)
        if np.isnan(size):
            # size = config.batch_size - len(tfdata)
            tfdata += get_dataset(data)
        else:
            # size = int(config.batch_size*ratio/ratios.sum())
            random_sample = np.random.choice(data, size=int(size))
            tfdata += get_dataset(random_sample)

    np.random.shuffle(tfdata)

    with tf.python_io.TFRecordWriter(str(tfrecord)) as writer:
        for i, sample in enumerate(tfdata):
            filename = h5file.parent.joinpath(sample['filename'])
            add_to_tfrecord(writer, filename, sample)

            if (i+1) % 100 == 0:
                print('\r{}/{} samples have been added to tfrecord file.'.format(i+1, len(tfdata)), end='')

    print('\rtfrecord file {} has been written, batch size is {}.'.format(tfrecord, len(tfdata)))


