# coding:utf-8
__author__ = 'Ruslan N. Kosarev'


# config to train P-Net (prediction net)
class PNetConfig:
    image_size = 12


# config to train R-Net (refinement net)
class RNetConfig:
    image_size = 24


# config to train O-Net (output net)
class ONetConfig:
    image_size = 48


class MTCNNConfig:
    pnet = PNetConfig()
    rnet = RNetConfig()
    onet = ONetConfig()

    min_face_size = 20
