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
    t = [i-t[0] for i in t] 
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

def calc_APD_change(data, base_label, purturbation, element): 
    initial_APD90 = calc_APD(np.array(eval(data['t_'+base_label][element])), np.array(eval(data['v_'+base_label][element])), 90)
    final_APD90 = calc_APD(np.array(eval(data['t_'+purturbation][element])), np.array(eval(data['v_'+purturbation][element])), 90)
    percent_change = ((final_APD90-initial_APD90)/(initial_APD90))*100
    return(percent_change)

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

#%% Assess APD90s
APDs_base = []
APDs_immune = []
for i in list(range(0, len(filtered_data['t']))):
    APD_b = calc_APD(np.array(eval(filtered_data['t'][i])), np.array(eval(filtered_data['v'][i])), 90)
    APD_i = calc_APD(np.array(eval(filtered_immune_data['t'][i])), np.array(eval(filtered_immune_data['v'][i])), 90)
    APDs_base.append(APD_b)
    APDs_immune.append(APD_i)

plt.hist(APDs_base, bins = 30, color = 'red', alpha = 0.5, label = 'Baseline Population')
plt.hist(APDs_immune, bins = 30, color = 'blue', alpha = 0.5, label = 'Profile with Profile Applied')
plt.legend()
plt.xlabel('APD 90')
plt.ylabel('Frequency')
plt.savefig(path + 'APD90_analysis.png')

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
#data = pd.read_csv(path+"data_1.csv")

data1 = pd.read_csv(path+"data_1.csv")
data2 = pd.read_csv(path+"data_2.csv")

data = data1.append(data2)
data = data.reset_index()

base_data = data.iloc[:, list(range(0,15))].copy(deep=False)
immune_data = data.iloc[:, [0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]].copy(deep=False)
immune_data = immune_data.rename(columns={'ind_i':'ind', 't_0_i': 't_0', 'v_0_i': 'v_0', 't_20_i': 't_20', 'v_20_i': 'v_20', 't_40_i': 't_40', 'v_40_i': 'v_40', 't_60_i': 't_60', 'v_60_i': 'v_60', 't_80_i':'t_80', 'v_80_i':'v_80', 't_100_i':'t_100', 'v_100_i': 'v_100'}) #must rename column names so plot_data() recognizes them

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
 
plt.plot(block, list(baseline.values()), '--ro', label = 'Baseline Data')
plt.plot(block, list(immunized.values()), '--bo', label = 'Immunized Data')
plt.legend()
plt.xlabel('IKr Block (%)')
plt.ylabel('Abnormal AP (%)')
plt.savefig(path+'abnormal_AP_quant')

# %% Quantify Data - each dictonary lists the percent change in APD90 for each IKr block

block = [0, 20, 40, 60, 80, 100]
baseline = {} 
immunized = {}

for i in list(range(0, len(block))):
    changes_base = []
    changes_imm = []

    for j in list(range(0,len(filtered_data['t_0']))):
        changes_base.append(calc_APD_change(filtered_data, str(block[0]), str(block[i]), j)) 
        changes_imm.append(calc_APD_change(filtered_immune_data, str(block[0]), str(block[i]), j))

    baseline[str(block[i]) + '%'] = np.mean(changes_base)
    immunized[str(block[i])+ '%'] = np.mean(changes_imm)
 
plt.plot(block, list(baseline.values()), '--ro', label = 'Baseline Data')
plt.plot(block, list(immunized.values()), '--bo', label = 'Immunized Data')
plt.legend()
plt.xlabel('IKr Block (%)')
plt.ylabel('Mean APD Change (%)')
plt.savefig(path+'apdchange_quant')


#%% 
block = [0, 20, 40, 60, 80, 100]
baseline = {} 
baseline_stds = {}
immunized = {}
immunized_stds = {}

for i in list(range(0, len(block))):
    APDs_base = []
    APDs_imm = []

    for j in list(range(0,len(filtered_data['t_0']))):
        APDs_base.append(calc_APD(np.array(eval(filtered_data['t_'+str(block[i])][j])), np.array(eval(filtered_data['v_'+str(block[i])][j])), 90)) 
        APDs_imm.append(calc_APD(np.array(eval(filtered_immune_data['t_'+str(block[i])][j])), np.array(eval(filtered_immune_data['v_'+str(block[i])][j])), 90)) 

    baseline[str(block[i]) + '%'] = np.mean(APDs_base)
    baseline_stds[str(block[i]) + '%'] = np.std(APDs_base)
    immunized[str(block[i])+ '%'] = np.mean(APDs_imm)
    immunized_stds[str(block[i]) + '%'] = np.std(APDs_imm)
 
plt.plot(block, list(baseline.values()), '--ro', label = 'Baseline Data', alpha = 0.5)
plt.plot(block, list(immunized.values()), '--bo', label = 'Immunized Data', alpha = 0.5)
plt.bar(block, list(baseline_stds.values()), color = 'red', alpha = 0.5, width = 5)
plt.bar(block, list(immunized_stds.values()), color = 'blue', alpha = 0.5, width = 5)


plt.legend()
plt.xlabel('IKr Block (%)')
plt.ylabel('Mean APD')
plt.savefig(path+'apd_quant')

# %% Compare RF APs to prolonged APs

APD90s = []
for i in list(range(0, len(filtered_immune_data['v_100']))):
    APD90 = calc_APD(np.array(eval(filtered_immune_data['t_100'][i])), np.array(eval(filtered_immune_data['v_100'][i])), 90)
    APD90s.append(APD90)

RF_APs = np.where(np.array(APD90s) == 1000.0)[0].tolist()
long_APs = np.where((np.array(APD90s) >= 905) & (np.array(APD90s) <= 960) == True)[0].tolist()
normal_APs = np.where((np.array(APD90s) >= 400) & (np.array(APD90s) <= 440) == True)[0].tolist()

fig, axs = plt.subplots(2, figsize=(15, 15))
for i in list(range(0, len(RF_APs))):
    axs[0].plot(eval(filtered_immune_data['t_100'][RF_APs[i]]), eval(filtered_immune_data['v_100'][RF_APs[i]]), color = 'red', alpha = 0.05*i, label = 'APs with RF')
    axs[0].plot(eval(filtered_immune_data['t_100'][long_APs[i]]), eval(filtered_immune_data['v_100'][long_APs[i]]), color = 'blue', alpha = 0.05*i, label = 'APs with prolonged Duration')
    axs[0].plot(eval(filtered_immune_data['t_100'][normal_APs[i]]), eval(filtered_immune_data['v_100'][normal_APs[i]]), color = 'green', alpha = 0.05*i, label = 'Normal APs')
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Voltage (mV)")

# Compare inds from RF APs vs long APs
for i in list(range(0, len(RF_APs))):
    axs[1].scatter(list(range(0,10)), list(eval(filtered_immune_data['ind'][RF_APs[i]]).values()), color = 'red', alpha = 0.05*i)
    axs[1].scatter(list(range(0,10)), list(eval(filtered_immune_data['ind'][long_APs[i]]).values()), color = 'blue', alpha = 0.05*i)
    axs[1].scatter(list(range(0,10)), list(eval(filtered_immune_data['ind'][normal_APs[i]]).values()), color = 'green', alpha = 0.05*i)

plt.sca(axs[1])
plt.xticks(range(10), ['ICaL', 'IKs', 'IKr', 'INaL', 'INa', 'Ito', 'IK1', 'INCX', 'INaK', 'IKb'])
axs[1].set_ylabel('Conductance')
plt.savefig(path+'conduct_analysis1')

#%% Look at profiles in which all 3 conditions have similar INaL
plt.scatter(list(range(0,10)), list(eval(filtered_immune_data['ind'][RF_APs[len(RF_APs)-1]]).values()), color = 'red', alpha = 0.05*len(RF_APs))
plt.scatter(list(range(0,10)), list(eval(filtered_immune_data['ind'][normal_APs[3]]).values()), color = 'green', alpha = 0.05*3)
plt.scatter(list(range(0,10)), list(eval(filtered_immune_data['ind'][long_APs[17]]).values()), color = 'blue', alpha = 0.05*17)
plt.xticks(list(range(0,10)), ['ICaL', 'IKs', 'IKr', 'INaL', 'INa', 'Ito', 'IK1', 'INCX', 'INaK', 'IKb'])
plt.ylabel('Conductance')
plt.savefig(path+'conduct_analysis2')

# %% Visualize Correlation between INaL and IKb
import seaborn as sn

RF_conds = []
long_conds = []
normal_conds = []

for i in list(range(0, len(RF_APs))):
    RF_conds.append(list(eval(filtered_immune_data['ind'][RF_APs[i]]).values()))
    long_conds.append(list(eval(filtered_immune_data['ind'][long_APs[i]]).values()))
    normal_conds.append(list(eval(filtered_immune_data['ind'][normal_APs[i]]).values()))

RF_data = pd.DataFrame(RF_conds, columns=['ICaL', 'IKs', 'IKr', 'INaL', 'INa', 'Ito', 'IK1', 'INCX', 'INaK', 'IKb'])
long_data = pd.DataFrame(long_conds, columns=['ICaL', 'IKs', 'IKr', 'INaL', 'INa', 'Ito', 'IK1', 'INCX', 'INaK', 'IKb'])
normal_data = pd.DataFrame(normal_conds, columns=['ICaL', 'IKs', 'IKr', 'INaL', 'INa', 'Ito', 'IK1', 'INCX', 'INaK', 'IKb'])

corrMatrix_RF = RF_data.corr()
corrMatrix_long = long_data.corr()
corrMatrix_normal = normal_data.corr()
#sn.heatmap(corrMatrix_RF, annot=True) #Visualize Corrlation Matrix

plt.scatter(RF_data['INaL'], RF_data['IKb'], color = 'red')
plt.scatter(long_data['INaL'], long_data['IKb'], color = 'blue')
plt.scatter(normal_data['INaL'], normal_data['IKb'], color = 'green')

m_rf, b_rf = np.polyfit(RF_data['INaL'], RF_data['IKb'], 1)
m_l, b_l = np.polyfit(long_data['INaL'], long_data['IKb'], 1)
m_n, b_n = np.polyfit(normal_data['INaL'], normal_data['IKb'], 1)

plt.plot(RF_data['INaL'], m_rf*RF_data['INaL']+b_rf, color = 'red', label = 'Pearson Correlation - RF APs: '+str(round(corrMatrix_RF['INaL']['IKb'],2)))
plt.plot(long_data['INaL'], m_l*long_data['INaL']+b_l, color = 'blue', label = 'Pearson Correlation - Long APs: '+str(round(corrMatrix_long['INaL']['IKb'],2)))
plt.plot(normal_data['INaL'], m_n*normal_data['INaL']+b_n, color = 'green', label = 'Pearson Correlation - Normal APs: '+str(round(corrMatrix_normal['INaL']['IKb'],2)))
plt.legend()
plt.xlabel('INaL Conductance')
plt.ylabel('IKb Conductance')
plt.savefig(path+'conduct_analysis3')

# %%
