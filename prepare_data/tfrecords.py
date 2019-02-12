# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import sys
import numpy as np
import tensorflow as tf
from prepare_data.tfrecord_utils import process_image_without_coder, _convert_to_example_simple
from prepare_data import h5utils

def add_to_tfrecord(writer, filename, data):
    image_data, height, width = process_image_without_coder(filename)
    example = _convert_to_example_simple(data, image_data)
    writer.write(example.SerializeToString())


def get_dataset(fname):

    positive = h5utils.read(fname, 'positive')
    negative = h5utils.read(fname, 'negative')
    part = h5utils.read(fname, 'part')
    landmark = h5utils.read(fname, 'landmark')

    positive = np.random.choice(positive, size=250000)
    negative = np.random.choice(negative, size=750000)
    part = np.random.choice(part, size=250000)
    # landmark = np.random.choice(landmark, size=250000)

    data = []

    for dataset in (positive, negative, part, landmark):
        for sample in dataset:
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

            data.append(row)

    # with fname.open() as f:
    #     data = []
    #
    #     for line in f.readlines():
    #         parts = line.strip().split(' ')
    #
    #         bbox = dict()
    #         bbox['xmin'] = 0
    #         bbox['ymin'] = 0
    #         bbox['xmax'] = 0
    #         bbox['ymax'] = 0
    #         bbox['xlefteye'] = 0
    #         bbox['ylefteye'] = 0
    #         bbox['xrighteye'] = 0
    #         bbox['yrighteye'] = 0
    #         bbox['xnose'] = 0
    #         bbox['ynose'] = 0
    #         bbox['xleftmouth'] = 0
    #         bbox['yleftmouth'] = 0
    #         bbox['xrightmouth'] = 0
    #         bbox['yrightmouth'] = 0
    #
    #         if len(parts) == 6:
    #             bbox['xmin'] = float(parts[2])
    #             bbox['ymin'] = float(parts[3])
    #             bbox['xmax'] = float(parts[4])
    #             bbox['ymax'] = float(parts[5])
    #         if len(parts) == 12:
    #             bbox['xlefteye'] = float(parts[2])
    #             bbox['ylefteye'] = float(parts[3])
    #             bbox['xrighteye'] = float(parts[4])
    #             bbox['yrighteye'] = float(parts[5])
    #             bbox['xnose'] = float(parts[6])
    #             bbox['ynose'] = float(parts[7])
    #             bbox['xleftmouth'] = float(parts[8])
    #             bbox['yleftmouth'] = float(parts[9])
    #             bbox['xrightmouth'] = float(parts[10])
    #             bbox['yrightmouth'] = float(parts[11])
    #
    #         row = dict()
    #         # row['filename'] = parts[0]
    #         row['filename'] = os.path.join(str(fname.parent), plib.Path(parts[0]).parent.name, plib.Path(parts[0]).name)
    #         row['label'] = int(parts[1])
    #         row['bbox'] = bbox
    #
    #         data.append(row)

    return data


def pnet_tfrecord(tfrecord, h5file, dataset_dir, shuffling=False, seed=None):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    np.random.seed(seed=seed)

    # tf record file name
    if tfrecord.exists():
        os.remove(str(tfrecord))

    # get data and shuffling.
    # fname = dataset_dir.joinpath('train_landmark.txt')
    dataset = get_dataset(h5file)

    # if shuffling:
    #     np.random.shuffle(dataset)

    with tf.python_io.TFRecordWriter(str(tfrecord)) as writer:
        for i, data in enumerate(dataset):
            filename = dataset_dir.joinpath(data['filename'])
            add_to_tfrecord(writer, filename, data)

            if (i+1) % 100 == 0:
                print('\r{}/{} images have been converted'.format(i+1, len(dataset)), end='')

    print('\nFinished converting the MTCNN dataset!')


