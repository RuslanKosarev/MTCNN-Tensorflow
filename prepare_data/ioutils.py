# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import cv2


def write_image(image, filename, prefix=None):
    if prefix is not None:
        filename = os.path.join(str(prefix), str(filename))

    if not cv2.imwrite(str(filename), image):
        raise IOError('while writing the file {}.'.format(filename))


def read_image(filename, prefix=None):
    if prefix is not None:
        image = cv2.imread(os.path.join(str(prefix), str(filename)))
    else:
        image = cv2.imread(str(filename))

    if image is None:
        raise IOError('while reading the file {}.'.format(filename))
    return image


class ImageLoader:
    def __iter__(self):
        return self

    def __init__(self, data, prefix=None, display=100):
        self.counter = -1
        self.data = data
        self.display = display
        self.limit = len(data)
        self.prefix = str(prefix)

    def __next__(self):
        if self.counter < self.limit:
            if self.prefix is not None:
                image = read_image(os.path.join(self.prefix, str(self.data[self.counter])))
            else:
                image = read_image(str(self.data[self.counter]))

            self.counter += 1
            if self.counter > 0 and self.counter % self.display == 0:
                print('\rnumber of performed iterations {}/{}'.format(self.counter, self.limit), end='')
            return image
        else:
            print('\rnumber of iterations {}'.format(self.limit), end='\n')
            raise StopIteration
