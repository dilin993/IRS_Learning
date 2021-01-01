import numpy as np
from channelmodels.channelmodel import ChannelModel
import os.path
import scipy.constants
import utils.comutils as comutils


class RayTracingChannelModel(ChannelModel):

    def __init__(self, rxdim, txdim, rxid, txid, rxpt, txpt, raytracing_output_path):
        ChannelModel.__init__(self, rxdim, txdim)
        fname = 'hmatrix.txSet{0:03d}.txPt{1:03d}.rxSet{2:03d}.rxPt{3:03d}.inst001.csv'.format(txid, txpt, rxid, rxpt)
        data = np.genfromtxt(os.path.join(raytracing_output_path, fname), delimiter=' ',
                             usecols=tuple(range(1, 2*rxdim + 1)))
        if data.ndim < 2:
            data = data.reshape((2*rxdim, txdim))
        else:
            data = data.T
        for i in range(rxdim):
            for j in range(txdim):
                self.H[i, j] = data[2*i, j] + 1j * data[2*i + 1, j]


class RayTracingLosChannelModel(ChannelModel):

    def __init__(self, rxnode, txnode, rxpt, txpt, raytracing_angle_output_path, fc, delta=0.5):
        ChannelModel.__init__(self, rxnode.antnum, txnode.antnum)
        fname = \
            'angles.txSet{0:03d}.txPt{1:03d}.rxSet{2:03d}.rxPt{3:03d}.txEl001.rxEl001.csv'.format(txnode.node_id, txpt,
                                                                                                  rxnode.node_id, rxpt)
        data = np.genfromtxt(os.path.join(raytracing_angle_output_path, fname), delimiter=' ')
        arg_max_p = np.argmax(data[:, 6])
        phi_arrival = data[arg_max_p, 2]
        theta_arrival = data[arg_max_p, 3]
        phi_departure = data[arg_max_p, 4]
        theta_departure = data[arg_max_p, 5]
        lambda1 = scipy.constants.speed_of_light / (fc * 1e9)
        d = comutils.distance(txnode, rxnode)
        a_tx = comutils.generate_array_response(txnode.nx, txnode.ny, txnode.nz, delta, theta_departure, phi_departure)
        a_rx = comutils.generate_array_response(rxnode.nx, rxnode.ny, rxnode.nz, delta, theta_arrival, phi_arrival)
        self.H = comutils.path_loss_los(d, fc) * np.exp(-1j * 2 * np.pi * d / lambda1) * np.matmul(a_rx.T, a_tx.conj())



