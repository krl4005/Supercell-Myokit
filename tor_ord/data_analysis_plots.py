#%% 
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from ast import literal_eval
import math
import myokit
from scipy.signal import find_peaks # pip install scipy

#%%
for i in list(range(1,len(means)+1)):
    sc = plt.scatter([i]*len(conductance_groups[0]), conductance_groups[i-1])

positions = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
label = ('GCaL', 'GKs', 'GKr', 'GNaL', 'GNa', 'Gto', 'GK1', 'GNCX', 'GNaK', 'Gkb')
plt.ylabel("Conductance Value")
plt.xticks(positions, label)

#%% CORRELATION ANALYSIS
import seaborn as sn
import matplotlib.pyplot as plt

corrMatrix = df_cond.corr()
print (corrMatrix)
sn.heatmap(corrMatrix, annot=True)
plt.show()