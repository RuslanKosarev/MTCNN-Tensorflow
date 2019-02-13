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

    for sample in inpdata:
        bbox = dict()
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['xlefteye'] = 0
        bbox['ylefteye'] = 0
        bbox['xrighteye'] = 0
        bbox['yrighteye'] = 0
        bbox['xnose'] = 0
        bbox['ynose'] = 0
        bbox['xleftmouth'] = 0
        bbox['yleftmouth'] = 0
        bbox['xrightmouth'] = 0
        bbox['yrightmouth'] = 0

        if len(sample) == 6:
            bbox['xmin'] = float(sample[2])
            bbox['ymin'] = float(sample[3])
            bbox['xmax'] = float(sample[4])
            bbox['ymax'] = float(sample[5])
        if len(sample) == 12:
            bbox['xlefteye'] = float(sample[2])
            bbox['ylefteye'] = float(sample[3])
            bbox['xrighteye'] = float(sample[4])
            bbox['yrighteye'] = float(sample[5])
            bbox['xnose'] = float(sample[6])
            bbox['ynose'] = float(sample[7])
            bbox['xleftmouth'] = float(sample[8])
            bbox['yleftmouth'] = float(sample[9])
            bbox['xrightmouth'] = float(sample[10])
            bbox['yrightmouth'] = float(sample[11])

        row = dict()
        row['filename'] = sample[0]
        row['label'] = int(sample[1])
        row['bbox'] = bbox
        outdata.append(row)

    return outdata


def pnet_tfrecord(tfrecord, h5file, outdir, seed=None):
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
            filename = outdir.joinpath(sample['filename'])
            add_to_tfrecord(writer, filename, sample)

            if (i+1) % 100 == 0:
                print('\r{}/{} samples have been written.'.format(i+1, len(tfdata)), end='')

    print('\rtfrecord file {} has been written, batch size is {}.'.format(tfrecord, config.batch_size))


