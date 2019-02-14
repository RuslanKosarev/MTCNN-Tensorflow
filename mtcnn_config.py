# coding:utf-8
__author__ = 'Ruslan N. Kosarev'


# config to train R-Net (refinement net)
class RNetConfig:
    def __init__(self, **kwargs):
        self.image_size = 24


# config to train O-Net (output net)
class ONetConfig:
    def __init__(self, **kwargs):
        self.image_size = 24
