import numpy as np
import pandas as pd
from simulations.irs_v2x_simulation import IRSV2XSimulation
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_position_simulation_2.csv')
print(data)


def plot_data(col1, col2, label):
    plt.plot(data[col2].iloc[:], linestyle=':', label='Without IRS')
    plt.plot(data[col1].iloc[:], linestyle=':', label='With IRS')
    plt.xlabel('Tx Point')
    plt.ylabel(label)
    plt.legend()
    plt.grid()
    plt.show()


plot_data(IRSV2XSimulation.COL_CAPACITY_IRS, IRSV2XSimulation.COL_CAPACITY_NIRS, 'Achievable Rate(bits/s/Hz)')