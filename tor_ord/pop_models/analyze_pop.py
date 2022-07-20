#%% Import functions and tools
from math import log10
import random
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


# %% Define Functions
def calc_APD(t, v, apd_pct):
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    repol_pot = max_p - apa * apd_pct/100
    idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
    apd_val = t[idx_apd+max_p_idx]
    return(apd_val) 

def check_physio(t,v):
    """
    Compares the simulation data for an individual to the baseline Tor-ORd values. The returned error value is a sum of the differences between the individual and baseline values.
    Returns
    ------
        error
    """
    t = np.array(t)
    v = np.array(v)

    ap_features = {}
    feature_targets = {'Vm_peak': [10, 55],
                    'dvdt_max': [100, 1000],
                    'apd40': [85, 320],
                    'apd50': [110, 430],
                    'apd90': [180, 440],
                    'triangulation': [50, 150],
                    'RMP': [-95, -80]}

    # Voltage/APD features#######################
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    dvdt_max = np.max(np.diff(v[0:30])/np.diff(t[0:30]))

    ap_features['Vm_peak'] = max_p
    ap_features['dvdt_max'] = dvdt_max

    for apd_pct in [40, 50, 90]:
        apd_val = calc_APD(t,v,apd_pct) 
        ap_features[f'apd{apd_pct}'] = apd_val
 
    ap_features['triangulation'] = ap_features['apd90'] - ap_features['apd40']
    ap_features['RMP'] = np.mean(v[len(v)-50:len(v)])

    error = 0
    for k, v in ap_features.items():
        if ((v < feature_targets[k][0]) or
                (v > feature_targets[k][1])):
            error += 1000 
    
    if error == 0:
        ap_features['valid?'] = 0 #AP is valid, can be kept
    else:
        ap_features['valid?'] = 1 #AP is no valid, should be eliminated

    return(ap_features)

def detect_RF(t,v):

    #find slopes
    slopes = []
    for i in list(range(0, len(v)-1)):
        m = (v[i+1]-v[i])/(t[i+1]-t[i])
        slopes.append(round(m, 1))

    #find times and voltages at which slope is 0
    zero_slopes = np.where(slopes == np.float64(0.0))[0].tolist()
    zero_slopes_idx = np.where(np.diff(zero_slopes)!=1)[0].tolist()
    zero_slopes_idx.append(len(zero_slopes)) #list must end with last index

    #pull out groups of zero slope (indexes)
    zero_groups = []
    zero_groups.append(zero_slopes[0:zero_slopes_idx[0]+1])
    for x in list(range(0,len(zero_slopes_idx)-1)):
        g = zero_slopes[zero_slopes_idx[x]+1:zero_slopes_idx[x+1]+1]
        zero_groups.append(g)

    #pull out groups of zero slopes (voltages and times)
    vol_pos = []
    tim_pos = []
    for y in list(range(0,len(zero_groups))):
        vol = []
        tim = []
        for z in zero_groups[y]:
            vol.append(v[z])
            tim.append(t[z])
        vol_pos.append(vol)
        tim_pos.append(tim) 


    #Find RF given the conditions (voltage<-70 & time>100)
    no_RF = []
    for k in list(range(0, len(vol_pos))):
        if np.mean(vol_pos[k]) < -70 and np.mean(tim_pos[k]) > 100:
            no_RF.append(tim_pos[k])
            no_RF.append(vol_pos[k])

    #Report EAD 
    if len(no_RF)==0:
        info = "Repolarization failure!"
        result = 1
    else:
        info = "normal repolarization - resting membrane potential from t=", no_RF[0][0], "to t=", no_RF[0][len(no_RF[0])-1]
        result = 0
    return result

def plot_data(data):
    AP_label = []

    fig, axs = plt.subplots(4, figsize=(15, 15))
    for i in list(range(0, len(data['t']))):

        # check for valid AP
        features = check_physio(eval(data['t'][i]), eval(data['v'][i]))
        AP_label.append(features['valid?'])

        # baseline data 
        axs[0].plot((eval(data['t'][i])), eval(data['v'][i]), color = 'red', alpha = 0.5)
        axs[0].set_ylabel("Voltage (mV)")
        axs[0].set_title("Baseline Data")

        # EAD data (stimulus)
        axs[1].plot((eval(data['t_ead'][i])), eval(data['v_ead'][i]), color = 'red', alpha = 0.5)
        axs[1].set_ylabel("Voltage (mV)")
        axs[1].set_title("EAD Analysis (Stimulus)")

        # EAD data (ical)
        axs[2].plot((eval(data['t_ical'][i])), eval(data['v_ical'][i]), color = 'red', alpha = 0.5)
        axs[2].set_ylabel("Voltage (mV)")
        axs[2].set_title("EAD Analysis (I_CaL enhancement)")

        # RF data (iKr)
        axs[3].plot((eval(data['t_rf'][i])), eval(data['v_rf'][i]), color = 'red', alpha = 0.5)
        axs[3].set_xlabel("Time (ms)")
        axs[3].set_ylabel("Voltage (mV)")
        axs[3].set_title("RF Analysis (IKr Block)")

    return(AP_label)

#%% read in data
data = pd.read_csv("data.csv")
models_dicts = pd.read_csv("models.csv")

immune_data = pd.read_csv("immune_data.csv")
immune_models = pd.read_csv("immune_models.csv")

#%% plot APs and record labels (0 - represents normal, 1 - represents abnormal AP)
AP_labels = plot_data(data)
print(AP_labels)

AP_immune = plot_data(immune_data)
print(AP_immune)


# %% Calculate Variance
vari = []
for j in list(range(0, len(list(models_dicts[0].values())))):
    con = []

    for i in list(range(0,len(models_dicts))):
        con.append(list(models_dicts[i].values())[j])

    vari.append(np.var(con))

print(vari)
