from channelmodels.raytracing import RayTracingChannelModel
from utils.dbutils import RayTracingDatabase
from simulations.irs_v2x_simulation import IRSV2XSimulation
import utils.comutils as utils
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = "C:\\Users\\dilin\\Documents\\Wireless insight\\irs_v2x_city - 2\\City Area\\"
DB_PATH = \
    'C:\\Users\\dilin\\Documents\\Wireless insight\\irs_v2x_city - 2\\City Area\\irs_v2x_city.City Area.sqlite'
BS_ID = 2
IRS_ID = 3
VEHICLE_ID = 5

db = RayTracingDatabase(DB_PATH)
vehicle_positions = db.get_tx_positions(VEHICLE_ID)
bs_position = db.get_rx_positions(BS_ID)[0]
irs_position = db.get_tx_positions(IRS_ID)[0]
num_points = len(vehicle_positions)-100
bs_antennas = (1, 4, 2)
irs_antennas = (16, 1, 16)
vehicle_antennas = (1, 1, 1)
p_dbm = 30

simulation = IRSV2XSimulation(BS_ID, bs_antennas, bs_position, IRS_ID, irs_antennas, irs_position, VEHICLE_ID,
                              vehicle_antennas, vehicle_positions, 10, -40, -3, 30, p_dbm, DATA_PATH, start_point=100,
                              num_points=num_points, early_stop=True, irs_discrete_levels=8, k_r=1, k_c=1,
                              ml_model='classifier.joblib')

data = simulation.simulate()
data.to_csv('data_position_simulation_2.csv')
