import numpy as np


class ChannelModel():

    def __init__(self, rxdim, txdim):
        self.rxdim = rxdim
        self.txdim = txdim
        self.H = np.zeros((rxdim, txdim), dtype=complex)

    def hmatrix(self):
        return self.H