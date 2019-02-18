# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

from train_models.mtcnn_model import *


# config to train P-Net (prediction net)
class Config:
    image_size = 48
    number_of_epochs = 30
    number_of_iterations = 10000
    lr = 0.001
    batch_size = 384