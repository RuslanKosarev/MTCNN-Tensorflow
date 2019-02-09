# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import sys
import numpy as np
import tensorflow as tf
from prepare_data.tfrecord_utils import process_image_without_coder, _convert_to_example_simple


def add_to_tfrecord(writer, filename, data):
    image_data, height, width = process_image_without_coder(filename)
    example = _convert_to_example_simple(data, image_data)
    writer.write(example.SerializeToString())


def get_dataset(fname):

    with fname.open() as f:
        data = []

        for line in f.readlines():
            parts = line.strip().split(' ')

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

            if len(parts) == 6:
                bbox['xmin'] = float(parts[2])
                bbox['ymin'] = float(parts[3])
                bbox['xmax'] = float(parts[4])
                bbox['ymax'] = float(parts[5])
            if len(parts) == 12:
                bbox['xlefteye'] = float(parts[2])
                bbox['ylefteye'] = float(parts[3])
                bbox['xrighteye'] = float(parts[4])
                bbox['yrighteye'] = float(parts[5])
                bbox['xnose'] = float(parts[6])
                bbox['ynose'] = float(parts[7])
                bbox['xleftmouth'] = float(parts[8])
                bbox['yleftmouth'] = float(parts[9])
                bbox['xrightmouth'] = float(parts[10])
                bbox['yrightmouth'] = float(parts[11])

            row = dict()
            row['filename'] = parts[0]
            row['label'] = int(parts[1])
            row['bbox'] = bbox

            data.append(row)

    return data


def generate(dataset_dir, output_dir, shuffling=False, seed=None):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    np.random.seed(seed=seed)

    if not output_dir.exists():
        output_dir.mkdir()

    # tf record file name
    tfrecord_fname = output_dir.joinpath('train_PNet_landmark.tfrecord')
    if tfrecord_fname.exists():
        os.remove(str(tfrecord_fname))

    # get data and shuffling.
    fname = dataset_dir.joinpath('train_landmark.txt')
    dataset = get_dataset(fname)

    if shuffling:
        np.random.shuffle(dataset)

    with tf.python_io.TFRecordWriter(str(tfrecord_fname)) as writer:
        for i, data in enumerate(dataset):
            if i+1 % 100 == 0:
                sys.stdout.write('\r{}/{} images have been converted'.format(i+1, len(dataset)))
            sys.stdout.flush()

            filename = dataset_dir.joinpath(data['filename'])
            add_to_tfrecord(writer, filename, data)

    print('\nFinished converting the MTCNN dataset!')


