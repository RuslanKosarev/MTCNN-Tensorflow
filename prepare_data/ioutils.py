# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import cv2


def write_image(image, filename):
    if not cv2.imwrite(str(filename), image):
        raise IOError('while writing the file {}.'.format(filename))
