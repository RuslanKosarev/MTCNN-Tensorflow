# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

from train_models.mtcnn_model import *


# config to train O-Net (output net)
class Config:
    def __init__(self):
        self.image_size = 12
        self.number_of_epochs = 20
        self.number_of_iterations = 10000
        self.lr = 0.01
        self.batch_size = 384

        self.pos_ratio = 1
        self.neg_ratio = 3
        self.part_ratio = 1
        self.landmark_ratio = 1

        self.factory = ONet


# construct ONet
class ONet:
    def __init__(self, inputs, label=None, bbox_target=None, landmark_target=None, training=True):
        pass
