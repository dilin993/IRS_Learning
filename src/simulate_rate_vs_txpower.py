from channelmodels.raytracing import RayTracingChannelModel
from utils.dbutils import RayTracingDatabase
from simulations.irs_v2x_simulation import IRSV2XSimulation
import utils.comutils as utils
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "C:\\Users\\dilin\\Documents\\Wireless insight\\irs_v2x_city\\City Area\\"
DB_PATH = \
    'C:\\Users\\dilin\\Documents\\Wireless insight\\irs_v2x_city\\City Area\\irs_v2x_city.City Area.sqlite'
BS_ID = 2
IRS_ID = 3
VEHICLE_ID = 4

db = RayTracingDatabase(DB_PATH)
vehicle_positions = db.get_tx_positions(VEHICLE_ID)
bs_position = db.get_rx_positions(BS_ID)[0]
irs_position = db.get_tx_positions(IRS_ID)[0]
num_points = 1
i = 0
p_dbm_all = np.linspace(0, 30, 100)
data_256 = None
data_64 = None
data_256_sub2 = None
data_64_sub2 = None
for p_dbm in p_dbm_all:
    simulation_256 = IRSV2XSimulation(BS_ID, (1, 4, 2), bs_position, IRS_ID, (16, 1, 16), irs_position, VEHICLE_ID,
                                      (1, 1, 1), vehicle_positions, 10, -40, -3, 30, p_dbm, DATA_PATH, start_point=15,
                                      num_points=num_points, early_stop=True, irs_discrete_levels=8, k_r=1, k_c=1)
    simulation_64 = IRSV2XSimulation(BS_ID, (1, 4, 2), bs_position, IRS_ID, (8, 1, 8), irs_position, VEHICLE_ID,
                                     (1, 1, 1), vehicle_positions,10, -40, -3, 30, p_dbm, DATA_PATH, start_point=15,
                                     num_points=num_points, early_stop=True, irs_discrete_levels=8, k_r=1, k_c=1)
    simulation_256_sub2 = IRSV2XSimulation(BS_ID, (1, 4, 2), bs_position, IRS_ID, (16, 1, 16), irs_position, VEHICLE_ID,
                                           (1, 1, 1), vehicle_positions, 10, -40, -3, 30, p_dbm, DATA_PATH,
                                           start_point=15, num_points=num_points, early_stop=True,
                                           irs_discrete_levels=8, k_r=1, k_c=1, enable_los_beamforming=True)
    simulation_64_sub2 = IRSV2XSimulation(BS_ID, (1, 4, 2), bs_position, IRS_ID, (8, 1, 8), irs_position, VEHICLE_ID,
                                          (1, 1, 1), vehicle_positions, 10, -40, -3, 30, p_dbm, DATA_PATH,
                                          start_point=15, num_points=num_points, early_stop=True, irs_discrete_levels=8,
                                          k_r=1, k_c=1, enable_los_beamforming=True)
    if data_256 is None:
        data_256 = simulation_256.simulate()
    else:
        data_256 = data_256.append(simulation_256.simulate())
    if data_256_sub2 is None:
        data_256_sub2 = simulation_256_sub2.simulate()
    else:
        data_256_sub2 = data_256_sub2.append(simulation_256_sub2.simulate())
    if data_64 is None:
        data_64 = simulation_64.simulate()
    else:
        data_64 = data_64.append(simulation_64.simulate())
    if data_64_sub2 is None:
        data_64_sub2 = simulation_64_sub2.simulate()
    else:
        data_64_sub2 = data_64_sub2.append(simulation_64_sub2.simulate())
data_256.to_csv('data_256.csv')
data_64.to_csv('data_64.csv')


def plot_data(col1, col2, label):
    plt.plot(p_dbm_all, data_256[col2].iloc[:], linestyle=':', label='Without IRS')
    plt.plot(p_dbm_all, data_256[col1].iloc[:], linestyle='-', label='With IRS(256) full CSI')
    plt.plot(p_dbm_all, data_64[col1].iloc[:], linestyle='-', label='With IRS(64) full CSI')
    plt.plot(p_dbm_all, data_256_sub2[col1].iloc[:], linestyle=':', label='With IRS(256) Los')
    plt.plot(p_dbm_all, data_64_sub2[col1].iloc[:], linestyle=':', label='With IRS(64) Los')
    plt.xlabel('Tx Power(dBm)')
    plt.ylabel(label)
    plt.legend()
    plt.grid()
    plt.show()


plot_data(IRSV2XSimulation.COL_CAPACITY_IRS, IRSV2XSimulation.COL_CAPACITY_NIRS, 'Achievable Rate(bits/s/Hz)')






