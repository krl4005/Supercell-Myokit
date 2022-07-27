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

def plot_data(data_frame, c, data_type, title):
    AP_label = []

    fig, axs = plt.subplots(len(data_type), figsize=(15, 15))
    for i in list(range(0, len(data_frame[data_type[0][0]]))):

        # Check for valid AP
        features = check_physio(eval(data_frame[data_type[0][0]][i]), eval(data_frame[data_type[0][1]][i]))
        AP_label.append(features['valid?'])

        # Plotting Data
        for p in list(range(0,len(data_type))):

            axs[p].plot((eval(data_frame[data_type[p][0]][i])), eval(data_frame[data_type[p][1]][i]), color = c, alpha = 0.5)
            axs[p].set_ylabel("Voltage (mV)")
            axs[p].set_title(title[p])
            if p == len(data_type):
                axs[p].set_xlabel('Time (ms)')

    return(AP_label)

def detect_EAD(t, v):
    #find slope
    slopes = []
    for i in list(range(0, len(v)-1)):
        m = (v[i+1]-v[i])/(t[i+1]-t[i])
        slopes.append(round(m, 2))

    #find rises
    pos_slopes = np.where(slopes > np.float64(0.0))[0].tolist()
    pos_slopes_idx = np.where(np.diff(pos_slopes)!=1)[0].tolist()
    pos_slopes_idx.append(len(pos_slopes)) #list must end with last index

    #pull out groups of rises (indexes)
    pos_groups = []
    pos_groups.append(pos_slopes[0:pos_slopes_idx[0]+1])
    for x in list(range(0,len(pos_slopes_idx)-1)):
        g = pos_slopes[pos_slopes_idx[x]+1:pos_slopes_idx[x+1]+1]
        pos_groups.append(g)

    #pull out groups of rises (voltages and times)
    vol_pos = []
    tim_pos = []
    for y in list(range(0,len(pos_groups))):
        vol = []
        tim = []
        for z in pos_groups[y]:
            vol.append(v[z])
            tim.append(t[z])
        vol_pos.append(vol)
        tim_pos.append(tim) 

    #Find EAD given the conditions (voltage>-70 & time>100)
    EADs = []
    EAD_vals = []
    for k in list(range(0, len(vol_pos))):
        if np.mean(vol_pos[k]) > -70 and np.mean(tim_pos[k]) > 100:
            EAD_vals.append(tim_pos[k])
            EAD_vals.append(vol_pos[k])
            EADs.append(max(vol_pos[k])-min(vol_pos[k]))

    #Report EAD 
    if len(EADs)==0:
        info = "no EAD"
        result = 0
    else:
        info = "EAD:", round(max(EADs))
        result = 1
    
    return result

def detect_RF(t, v):

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

def detect_APD(t, v, apd90_base):
    APD90_i = calc_APD(t, v, 90)
    APD90_error = (APD90_i - apd90_base)/(APD90_i)*100
    if APD90_error < 40:
        result_APD = 0
    else:
        result_APD = 1
    return(result_APD)

def calculate_variance(d):
    vari = []
    for j in list(range(0, len(eval(d['ind'][0])))):
        con = []

        for i in list(range(0,len(d['ind']))):
            con.append(list(eval(d['ind'][i]).values())[j])

        vari.append(np.var(con))

    return(vari) 

def count_abnormal_APs(data_frame, data_type):
    result = []

    for i in list(range(0, len(data_frame['t_' + data_type]))):
        time_point = eval(data_frame['t_' + data_type].tolist()[0])[0]
        t = [e-time_point for e in eval(data_frame['t_' + data_type].tolist()[i])]
        v = [eval(data_frame['v_' + data_type].tolist()[i])][0]

        EAD = detect_EAD(t,v)
        RF = detect_RF(t,v)
        features = check_physio(t, v)

        if EAD==1 or RF==1: #or features['apd90']>440: #or features['Vm_peak']<10:
            result.append(1)
        else:
            result.append(0)
    return(result)

########################################## ANALYSIS 1 #######################################################
#%% read in data
path = 'c:\\Users\\Kristin\\Desktop\\pop_models\\analysis1\\'
data = pd.read_csv(path+"data.csv")

base_data = data.iloc[:, list(range(0,10))].copy(deep=False)
immune_data = data.iloc[:, [0, 10, 11, 12, 13, 14, 15, 16, 17, 18]].copy(deep=False)
immune_data = immune_data.rename(columns={'t_imm': 't', 'v_imm': 'v', 't_ead_imm': 't_ead', 'v_ead_imm': 'v_ead', 't_ical_imm': 't_ical', 'v_ical_imm': 'v_ical', 't_rf_imm': 't_rf', 'v_rf_imm': 'v_rf'}) #must rename column names so plot_data() recognizes them


#%% plot APs and record labels (0 - represents normal, 1 - represents abnormal AP)

AP_labels = plot_data(base_data, 'red', [['t', 'v'], ['t_ical', 'v_ical'], ['t_rf', 'v_rf']], ["Baseline Data", "EAD Analysis (I_CaL enhancement)", "RF Analysis (IKr Block)"])
print(len(AP_labels))

AP_immune = plot_data(immune_data, 'blue', [['t', 'v'], ['t_ical', 'v_ical'], ['t_rf', 'v_rf']], ["Baseline Data", "EAD Analysis (I_CaL enhancement)", "RF Analysis (IKr Block)"])

#%% filter data
filtered_data = base_data.copy(deep=False)
filtered_immune_data = immune_data.copy(deep=False)

ind_to_drop = []
for i in list(range(0,len(AP_labels))):
    if AP_labels[i] == 1:
        ind_to_drop.append(i)

filtered_data.drop(ind_to_drop, axis = 0, inplace=True)
filtered_data = filtered_data.reset_index()
filtered_immune_data.drop(ind_to_drop, axis = 0, inplace=True)
filtered_immune_data = filtered_immune_data.reset_index()

#%% plot filtered data
filtered_AP_labels = plot_data(filtered_data, 'red', [['t', 'v'], ['t_ical', 'v_ical'], ['t_rf', 'v_rf']], ["Baseline Data", "EAD Analysis (I_CaL enhancement)", "RF Analysis (IKr Block)"])
plt.savefig(path + 'filtered_baseline.png')
print(filtered_AP_labels)

filtered_AP_immune = plot_data(filtered_immune_data, 'blue', [['t', 'v'], ['t_ical', 'v_ical'], ['t_rf', 'v_rf']], ["Baseline Data", "EAD Analysis (I_CaL enhancement)", "RF Analysis (IKr Block)"])
plt.savefig(path + 'filtered_immunized.png')

#%% Quantify EAD base
total_APs = len(filtered_data["t_ical"])

abnormal_baseline_ical = count_abnormal_APs(filtered_data, 'ical')
print('percent of abnormal APs after ical enhancement:', (abnormal_baseline_ical.count(1)/total_APs)*100, '%')

abnormal_immune_ical = count_abnormal_APs(filtered_immune_data, 'ical')
print('percent of abnormal APs after immunization with ical enhancement:', (abnormal_immune_ical.count(1)/total_APs)*100, '%')

abnormal_baseline_rf = count_abnormal_APs(filtered_data, 'rf')
print('percent of abnormal APs after ikr block:', (abnormal_baseline_rf.count(1)/total_APs)*100, '%')

abnormal_immune_rf = count_abnormal_APs(filtered_immune_data, 'rf')
print('percent of abnormal APs after immunization with ikr block:', (abnormal_immune_rf.count(1)/total_APs)*100, '%')

print('percent improvement ical:', ((abnormal_immune_ical.count(1) - abnormal_baseline_ical.count(1))/abnormal_baseline_ical.count(1))*100)
print('percent improvement rf:', ((abnormal_immune_rf.count(1) - abnormal_baseline_rf.count(1))/abnormal_baseline_rf.count(1))*100)


########################################## ANALYSIS 2 #######################################################
#%% Read in Data
path = 'c:\\Users\\Kristin\\Desktop\\pop_models\\analysis2\\'
data1 = pd.read_csv(path+"data_1.csv")
data2 = pd.read_csv(path+"data_2.csv")

data = data1.append(data2)
data = data.reset_index()

base_data = data.iloc[:, list(range(0,14))].copy(deep=False)
immune_data = data.iloc[:, [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]].copy(deep=False)
immune_data = immune_data.rename(columns={'t_base_i': 't_0', 'v_base_i': 'v_0', 't_20_i': 't_20', 'v_20_i': 'v_20', 't_40_i': 't_40', 'v_40_i': 'v_40', 't_60_i': 't_60', 'v_60_i': 'v_60', 't_80_i':'t_80', 'v_80_i':'v_80', 't_100_i':'t_100', 'v_100_i': 'v_100'}) #must rename column names so plot_data() recognizes them

# %%
AP_labels = plot_data(base_data, 'red', [['t_0', 'v_0'], ['t_20', 'v_20'], ['t_40', 'v_40'], ['t_60', 'v_60'], ['t_80', 'v_80'], ['t_100', 'v_100']], ["Baseline Data", "20% IKr Block", "40% IKr block", "60% IKr block", "80% IKr block", "100% IKr block"])
print(len(AP_labels))

AP_immune = plot_data(immune_data, 'blue', [['t_0', 'v_0'], ['t_20', 'v_20'], ['t_40', 'v_40'], ['t_60', 'v_60'], ['t_80', 'v_80'], ['t_100', 'v_100']], ["Baseline Data", "20% IKr Block", "40% IKr block", "60% IKr block", "80% IKr block", "100% IKr block"])

# %% filter data
filtered_data = base_data.copy(deep=False)
filtered_immune_data = immune_data.copy(deep=False)

ind_to_drop = []
for i in list(range(0,len(AP_labels))):
    if AP_labels[i] == 1:
        ind_to_drop.append(i)

filtered_data.drop(ind_to_drop, axis = 0, inplace=True)
filtered_data = filtered_data.reset_index()
filtered_immune_data.drop(ind_to_drop, axis = 0, inplace=True)
filtered_immune_data = filtered_immune_data.reset_index()

#%% Plot filtered data
filtered_AP_labels = plot_data(filtered_data, 'red', [['t_0', 'v_0'], ['t_20', 'v_20'], ['t_40', 'v_40'], ['t_60', 'v_60'], ['t_80', 'v_80'], ['t_100', 'v_100']], ["Baseline Data", "20% IKr Block", "40% IKr block", "60% IKr block", "80% IKr block", "100% IKr block"])
plt.savefig(path + 'filtered_baseline.png')
total_APs = len(filtered_AP_labels)
print(total_APs)

filtered_AP_immune = plot_data(filtered_immune_data, 'blue', [['t_0', 'v_0'], ['t_20', 'v_20'], ['t_40', 'v_40'], ['t_60', 'v_60'], ['t_80', 'v_80'], ['t_100', 'v_100']], ["Baseline Data", "20% IKr Block", "40% IKr block", "60% IKr block", "80% IKr block", "100% IKr block"])
plt.savefig(path + 'filtered_immunized.png')

# %% Quantify Data - each dictonary lists the percent of abnormal APs for each IKr block

block = [0, 20, 40, 60, 80, 100]
baseline = {} 
immunized = {}

for i in list(range(0, len(block))):
    count_baseline = count_abnormal_APs(filtered_data, str(block[i]))
    baseline[str(block[i]) + '%'] = (count_baseline.count(1)/total_APs)*100

    count_immunized = count_abnormal_APs(filtered_immune_data, str(block[i]))
    immunized[str(block[i])+ '%'] = (count_immunized.count(1)/total_APs)*100

print('baseline:', baseline)
print('')
print('immunized', immunized)

#%%
plt.plot(block, list(baseline.values()), '--ro', label = 'Baseline Data')
plt.plot(block, list(immunized.values()), '--bo', label = 'Immunized Data')
plt.legend()
plt.xlabel('IKr Block (%)')
plt.ylabel('Abnormal AP (%)')
plt.savefig(path+'abnormal_AP_quant')
# %%
