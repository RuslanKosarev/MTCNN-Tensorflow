# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
import h5py
import numpy as np

lfwdtype = np.dtype([('path', h5py.special_dtype(vlen=str)), ('label', np.int8),
                     ('1', np.float), ('2', np.float), ('3', np.float), ('4', np.float), ('5', np.float),
                     ('6', np.float), ('7', np.float), ('8', np.float), ('9', np.float), ('10', np.float)])


dftdir = plib.Path(os.pardir).joinpath('data').absolute()


# ======================================================================================================================
class LFWDBase:
    def __init__(self, netconfig=None, inpdir=None, outdir=None, h5file=None):
        # input directory to LFW database
        if inpdir is None:
            inpdir = dftdir.joinpath('lfw')
        self.inpdir = plib.Path(inpdir).absolute()
        self.annotations = self.inpdir.joinpath('trainImageList.txt')

        # output directory for LFW database
        self.dtype = lfwdtype

        if outdir is None:
            if type(netconfig) is PNetConfig:
                self.outdir = dftdir.joinpath('pnet')
            elif type(netconfig) is RNetConfig:
                self.outdir = dftdir.joinpath('rnet')
            elif type(netconfig) is ONetConfig:
                self.outdir = dftdir.joinpath('onet')
            else:
                raise IOError('net config in not defined')
        else:
            self.outdir = plib.Path(outdir).absolute()

        if h5file is None:
            self.h5file = self.outdir.joinpath('dbtrain.h5')
        else:
            self.h5file = self.outdir.joinpath(h5file)

        self.keys = ('landmark',)
        for key in self.keys:
            if not self.outdir.joinpath(key).exists():
                self.outdir.joinpath(key).mkdir(parents=True)


# config to train P-Net (prediction net)
class PNetConfig:
    image_size = 12


# ======================================================================================================================
# config to train R-Net (refinement net)
class RNetConfig:
    image_size = 24
    number_of_epochs = 30
    number_of_iterations = 10000
    lr = 0.001
    batch_size = 500


# ======================================================================================================================
# config to train O-Net (output net)
class ONetConfig:
    image_size = 48


class MTCNNConfig:
    pnet = PNetConfig()
    rnet = RNetConfig()
    onet = ONetConfig()

    min_face_size = 20
