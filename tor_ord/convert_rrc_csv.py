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
from multiprocessing import Pool
import time

#%%
path = 'C:\\Users\\Kristin\\Desktop\\iter4\\g100_p200_e1_binarySearch\\12hrRun\\trial1\\'
rrcs = pd.read_csv('RRCs.csv')
rr = pd.DataFrame()
s = 0

rr['gen1'] = rrcs.loc[200:499].values.flatten().tolist()
for i in list(range(2,99)):
    label = 'gen' + str(i)
    s += 200
    print(s)
    rr[label] = rrcs.loc[200+s:499+s].values.flatten().tolist()
    rr.to_csv(path + 'RR.csv', index=False)

#rr.to_csv(path + 'RR.csv', index=False)

# %%
