# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import pathlib as plib
from datetime import datetime
from prepare_data import prepare
import config


if __name__ == '__main__':
    seed = None

    # prepare lfw database
    netconfig = config.RNetConfig()
    lfwdbase = config.LFWDBase(netconfig=netconfig)

    start = datetime.now()
    prepare.lfwdbase(dbase=lfwdbase, netconfig=netconfig, seed=seed)
    print(datetime.now() - start)
