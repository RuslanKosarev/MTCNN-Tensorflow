# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import cv2


def write_image(image, filename):
    if not cv2.imwrite(str(filename), image):
        raise IOError('while writing the file {}.'.format(filename))


def read_image(filename):
    image = cv2.imread(str(filename))
    if image is None:
        raise IOError('while reading the file {}.'.format(filename))
    return image


class ImageLoader:
    def __iter__(self):
        return self

    def __init__(self, data, display=3):
        self.counter = -1
        self.data = data
        self.display = display
        self.limit = data

    def __next__(self):
        if self.counter is not self.data:
            # image = read_image(self.data[self.counter])
            self.counter += 1
            if (self.counter + 1) % self.display == 0:
                print('\rnumber of performed iterations {}/{}.'.format(self.counter, self.limit, end=''))
            return self.counter
        else:
            raise StopIteration
