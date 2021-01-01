from simulations.simulation import Simulation
from nodes.node import BS, IRS, Vehicle
import utils.comutils as utils
from nodes.links import RayTracingLinks, RayTracingLosLinks
import numpy as np
import pandas as pd
import utils.arrayutils as arrayutils
from joblib import load


class IRSV2XSimulation(Simulation):
    COL_POS_X = 'x'
    COL_POS_Y = 'y'
    COL_POS_Z = 'z'
    COL_SPEED = 'speed'
    COL_OUTAGE_IRS = 'outage probability with irs'
    COL_GAIN_IRS = 'channel gain with irs'
    COL_CAPACITY_IRS = 'capacity with irs'
    COL_OUTAGE_NIRS = 'outage probability without irs'
    COL_GAIN_NIRS = 'channel gain without irs'
    COL_CAPACITY_NIRS = 'capacity without irs'
    COL_PHASE = 'phase_'

    def __init__(self, id_bs, antennas_bs, pos_bs, id_irs, antennas_irs, pos_irs, id_v, antennas_v, pos_v, speed_v,
                 sigma_n_sqr_dbm, sigma_sqr_db, gamma_th_db, p_dbm, data_path, irs_discrete_levels=16, start_point=1,
                 num_points=1, error_threshold=1e-3, k_c=1, k_r=1, max_iterations=100, early_stop=False, fc=24.2,
                 delta=0.5, enable_los_beamforming=False, ml_model=None):
        self.bs = BS(antennas_bs, pos_bs, id_bs)
        self.angles = np.linspace(0, 2*np.pi, irs_discrete_levels)
        self.irs = IRS(antennas_irs, pos_irs, id_irs)
        self.n_r_irs = self.irs.nz
        if self.irs.nx > 1:
            self.n_c_irs = self.irs.nx
        else:
            self.n_c_irs = self.irs.ny
        n_irs = self.n_r_irs * self.n_c_irs
        self.vehicle_positions = pos_v
        self.vehicle = Vehicle(antennas_v, (0, 0, 0), id_v, speed_v)
        self.sigma_n_sqr = utils.db2lin(sigma_n_sqr_dbm - 30)
        self.sigma_sqr = utils.db2lin(sigma_sqr_db)
        self.p_sig = utils.db2lin(p_dbm - 30)
        self.gamma_th = utils.db2lin(gamma_th_db)
        self.start = start_point
        self.error_threshold = error_threshold
        self.links = RayTracingLinks(data_path + 'hmatrix')
        self.links.add_link(self.vehicle, self.bs, start_point, 1)
        self.links.add_link(self.vehicle, self.irs, start_point, 1)
        self.links.add_link(self.irs, self.bs, 1, 1)
        self.num_points = num_points
        self.max_iterations = max_iterations
        self.early_stop = early_stop
        self.k_c = k_c
        self.k_r = k_r
        self.fc = fc
        self.delta = 0.5
        self.enable_los_beamforming = enable_los_beamforming
        if self.enable_los_beamforming:
            self.loslinks = RayTracingLosLinks(data_path + 'doa', fc, delta)
            self.loslinks.add_link(self.vehicle, self.bs, start_point, 1)
            self.loslinks.add_link(self.vehicle, self.irs, start_point, 1)
            self.loslinks.add_link(self.irs, self.bs, 1, 1)
        self.ml_model = ml_model
        if self.ml_model is not None:
            self.classifier = load(ml_model)

    def get_codebook_size(self):
        return self.codebook.shape[0]

    def simulate(self):
        index = self.start
        cols = [IRSV2XSimulation.COL_POS_X,
                IRSV2XSimulation.COL_POS_Y,
                IRSV2XSimulation.COL_POS_Z,
                IRSV2XSimulation.COL_SPEED,
                IRSV2XSimulation.COL_OUTAGE_IRS,
                IRSV2XSimulation.COL_GAIN_IRS,
                IRSV2XSimulation.COL_CAPACITY_IRS,
                IRSV2XSimulation.COL_OUTAGE_NIRS,
                IRSV2XSimulation.COL_GAIN_NIRS,
                IRSV2XSimulation.COL_CAPACITY_NIRS]
        for n in range(self.irs.antnum):
            cols.append(IRSV2XSimulation.COL_PHASE + str(n))
        output = pd.DataFrame(columns=cols)
        i = 0
        while i < self.num_points:
            pos = self.vehicle_positions[i]
            car = Vehicle(self.vehicle.get_antennas(), pos, self.vehicle.node_id, self.vehicle.speed)
            output = output.append(pd.Series(0, index=output.columns), ignore_index=True)
            output[IRSV2XSimulation.COL_POS_X].iloc[i] = pos[0]
            output[IRSV2XSimulation.COL_POS_Y].iloc[i] = pos[1]
            output[IRSV2XSimulation.COL_POS_Z].iloc[i] = pos[2]
            output[IRSV2XSimulation.COL_SPEED].iloc[i] = car.speed
            if index > self.start:
                self.links.modify_link(car, self.bs, index, 1)
                self.links.modify_link(car, self.irs, index, 1)
                if self.enable_los_beamforming:
                    self.loslinks.modify_link(car, self.bs, index, 1)
                    self.loslinks.modify_link(car, self.irs, index, 1)
            if self.ml_model is None:
                v, phase_indexes = self.subgroup_discrete_optimization(self.get_sumrate, minimization=False)
            else:
                x = np.array([pos[0], pos[1]]).reshape(1, -1)
                phase_indexes = self.classifier.predict(x)
                phase_indexes = phase_indexes.reshape(-1, 1)
                theta = np.zeros((self.irs.antnum, 1))
                for n in range(self.irs.antnum):
                    theta[n, 0] = self.angles[phase_indexes[n, 0]-1]
                v = np.exp(1j * theta)
            outage_irs, outage_nirs = self.get_outage_probability(v)
            gain_irs, gain_nirs = self.get_channel_gains(v)
            capacity_irs, capacity_nirs = self.get_sumrate(v)
            output[IRSV2XSimulation.COL_OUTAGE_IRS].iloc[i] = outage_irs
            output[IRSV2XSimulation.COL_GAIN_IRS].iloc[i] = gain_irs
            output[IRSV2XSimulation.COL_CAPACITY_IRS].iloc[i] = capacity_irs
            output[IRSV2XSimulation.COL_OUTAGE_NIRS].iloc[i] = outage_nirs
            output[IRSV2XSimulation.COL_GAIN_NIRS].iloc[i] = gain_nirs
            output[IRSV2XSimulation.COL_CAPACITY_NIRS].iloc[i] = capacity_nirs
            for n in range(self.irs.antnum):
                output[IRSV2XSimulation.COL_PHASE + str(n)].iloc[i] = phase_indexes[n]
            print('Simulating vehicle position ', pos)
            print(output.iloc[0, :])
            print('\n')
            i = i + 1
            index = index + 1
        return output

    def get_channel_vectors(self, shifts, h_d=None, h_r=None, h_v=None):
        theta = np.diag(shifts[:, 0])
        if h_d is None:
            h_d = self.links.get_link(self.vehicle, self.bs).hmatrix()
        if h_r is None:
            h_r = self.links.get_link(self.irs, self.bs).hmatrix()
        if h_v is None:
            h_v = self.links.get_link(self.vehicle, self.irs).hmatrix()
        g = h_d + np.matmul(h_r, np.matmul(theta, h_v))
        return g, h_d

    def get_outage_probability(self, shifts, h_d=None, h_r=None, h_v=None):
        g, h_d = self.get_channel_vectors(shifts, h_d, h_r, h_v)
        p_out_nirs = utils.calculate_rician_outage_probability(h_d, self.sigma_sqr, self.sigma_n_sqr, self.gamma_th,
                                                               self.p_sig, self.bs.antnum)
        p_out_irs = utils.calculate_rician_outage_probability(g, self.sigma_sqr, self.sigma_n_sqr, self.gamma_th,
                                                              self.p_sig, self.bs.antnum)
        return p_out_irs, p_out_nirs

    def get_channel_gains(self, shifts, h_d=None, h_r=None, h_v=None):
        g, h_d = self.get_channel_vectors(shifts, h_d, h_r, h_v)
        return np.square(np.linalg.norm(g)), np.square(np.linalg.norm(h_d))

    def get_sumrate(self, shifts, h_d=None, h_r=None, h_v=None):
        g, h_d = self.get_channel_vectors(shifts, h_d, h_r, h_v)
        return utils.calculate_sumrate(g, self.sigma_n_sqr, self.p_sig), \
               utils.calculate_sumrate(h_d, self.sigma_n_sqr, self.p_sig)

    def discrete_optimizatiom(self, obj_function, h_d, h_r, h_v, minimization=True):
        Phi = np.matmul(h_r, np.diag(h_v[:, 0]))
        A = np.matmul(Phi.conj().T, Phi)
        b = np.matmul(Phi.conj().T, h_d)
        theta = np.zeros((h_v.shape[0], 1), dtype=complex)
        phase_indexes = np.zeros((h_v.shape[0], 1), dtype=int)
        for n in range(h_v.shape[0]):
            theta[n, 0] = self.angles[0]
            phase_indexes[n] = 1
        v = np.exp(1j * theta)
        obj_val, x = obj_function(v, h_d, h_r, h_v)
        if minimization:
            prev_obj_value = float('inf')
        else:
            prev_obj_value = float('-inf')
        for i in range(self.max_iterations):
            for n in range(h_v.shape[0]):
                tau_n = b[n, 0]
                for j in range(h_v.shape[0]):
                    if j != n:
                        tau_n = tau_n + A[n, j] * v[j]
                theta_n = np.angle(tau_n)
                min_diff = 1000
                phase_index = 1
                for ang in self.angles:
                    alpha = np.angle(np.exp(1j*ang))
                    diff = np.abs(alpha - theta_n)
                    if diff < min_diff:
                        min_diff = diff
                        theta[n, 0] = ang
                        phase_indexes[n] = phase_index
                    phase_index = phase_index + 1
            v = np.exp(-1j * theta)
            prev_obj_value = obj_val
            obj_val, x = obj_function(v, h_d, h_r, h_v)
            if self.early_stop and np.abs(obj_val - prev_obj_value) < self.error_threshold:
                break
        v = np.exp(1j * theta)
        return v, phase_indexes

    def subgroup_discrete_optimization(self, obj_function, minimization=True):
        if not self.enable_los_beamforming:
            h_d = self.links.get_link(self.vehicle, self.bs).hmatrix()
            h_r = self.links.get_link(self.irs, self.bs).hmatrix()
            h_v = self.links.get_link(self.vehicle, self.irs).hmatrix()
        else:
            h_d = self.loslinks.get_link(self.vehicle, self.bs).hmatrix()
            h_r = self.loslinks.get_link(self.irs, self.bs).hmatrix()
            h_v = self.loslinks.get_link(self.vehicle, self.irs).hmatrix()
        if self.k_c == 1 and self.k_r == 1:
            return self.discrete_optimizatiom(obj_function, h_d, h_r, h_v, minimization)
        h_r_exp = np.zeros((self.n_r_irs, self.n_c_irs, self.bs.antnum), dtype=complex)
        for i in range(self.bs.antnum):
            h_r_exp[:, :, i] = np.reshape(h_r[i, :], (self.n_r_irs, self.n_c_irs))
        h_v_exp = np.reshape(h_v, (self.n_r_irs, self.n_c_irs))
        h_r1 = np.zeros((self.bs.antnum, int(self.irs.antnum/(self.k_r * self.k_c))), dtype=complex)
        for i in range(self.bs.antnum):
            h_r1[i, :] = np.reshape(h_r_exp[0:-1:self.k_r, 0:-1:self.k_c, i],
                                    (int(self.irs.antnum/(self.k_r * self.k_c)), 1))[:, 0]
        h_v1 = np.reshape(h_v_exp[0:-1:self.k_r, 0:-1:self.k_c], (int(self.irs.antnum/(self.k_r * self.k_c)), 1))
        v1, phase_indexes = self.discrete_optimizatiom(obj_function, h_d, h_r1, h_v1, minimization)
        v1_exp = np.reshape(v1, (int(self.n_r_irs/self.k_r), int(self.n_c_irs/self.k_c)))
        phase_indexes = np.reshape(phase_indexes, (int(self.n_r_irs / self.k_r), int(self.n_c_irs / self.k_c)))
        return np.reshape(arrayutils.resize_array(v1_exp, self.n_r_irs, self.n_c_irs), (self.irs.antnum, 1)), \
               np.reshape(arrayutils.resize_array(phase_indexes, self.n_r_irs, self.n_c_irs), (self.irs.antnum, 1))

    def codebook_optimization(self, obj_function, minimization=True):
        pass
