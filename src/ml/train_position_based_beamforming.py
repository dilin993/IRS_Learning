import numpy as np
import pandas as pd
from simulations.irs_v2x_simulation import IRSV2XSimulation
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load

data = pd.read_csv('data_position_simulation.csv')
irs_antnum = 256

cols_x = [IRSV2XSimulation.COL_POS_X,
        IRSV2XSimulation.COL_POS_Y]
        # IRSV2XSimulation.COL_POS_Z,
        # IRSV2XSimulation.COL_SPEED]
cols_y = []
for n in range(irs_antnum):
    cols_y.append(IRSV2XSimulation.COL_PHASE + str(n))
X = data[cols_x].to_numpy()
Y = data[cols_y].to_numpy()
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

forest = RandomForestClassifier(n_estimators=100, random_state=1)
classifier = MultiOutputClassifier(forest, n_jobs=-1)
classifier.fit(X[0:100, :], Y[0:100, :])
dump(classifier, 'classifier.joblib')
print(classifier.score(X, Y))
