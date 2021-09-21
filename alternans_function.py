
#%% FUNCTIONS
from re import M
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 

def evaluate_APD90(dat): 
    i_stim=np.array(dat['stimulus.i_stim'].tolist())
    AP_S_where = np.where(i_stim!=0)[0]
    AP_S_diff = np.where(np.diff(AP_S_where)!=1)[0]+1
    peaks = AP_S_where[AP_S_diff]

    v = dat['membrane.v']
    t = dat['engine.time']

    AP = np.split(v, peaks)
    t1 = np.split(t, peaks)

    APD = []

    for n in list(range(0, len(AP))):
        mdp = min(AP[n])
        max_p = max(AP[n])
        max_p_idx = np.argmax(AP[n])
        apa = max_p - mdp

        repol_pot = max_p - (apa * 90/100)
        idx_apd = np.argmin(np.abs(AP[n][max_p_idx:] - repol_pot))
        apd_val = t1[n][idx_apd+max_p_idx]-t1[n][0]
        APD.insert(n, apd_val) 

    return(APD)


# %%

def calulate_APD(v_dat, t_dat):
    t_depol = v_dat[0].index(max(v_dat[0]))

    v_60 = []
    wrong = []
    for i in list(range(t_depol, len(v_dat[0]))):
        d = np.abs(-60-v_dat[0][i])
        if d < 1:
            v_60.append(i)
        else:
            wrong.append(i)

    t_index = v_60[0]
    t_repol = t_dat[t_index] 

    APD = t_repol - t_depol


