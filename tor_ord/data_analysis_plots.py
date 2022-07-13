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
import seaborn as sn

#%% READ IN DATA

path = 'c:\\Users\\Kristin\\Desktop\\iter4\\g100_p200_e2\\trial10'
#error_thres = 2000

#df_rrcs = pd.read_csv(path + '\\RRCs.csv')
#rrcs = df_rrcs.to_numpy().tolist()

df_chal = pd.read_csv(path + '\\challenges.csv')
challenges = df_chal.to_numpy().tolist()

df_best_pop = pd.read_csv(path + '\\best_conds.csv')
best_ind = df_best_pop.to_numpy().tolist()

df_best_error = pd.read_csv(path + '\\best_error.csv')
best_error = df_best_error.to_numpy().tolist()


#%% GA 3

def get_ind_data(ind):
    mod, proto, x = myokit.load('./tor_ord_endo2.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto

def get_feature_errors(t,v,cai,i_ion):
    """
    Compares the simulation data for an individual to the baseline Tor-ORd values. The returned error value is a sum of the differences between the individual and baseline values.
    Returns
    ------
        error
    """

    ap_features = {}

    # Returns really large error value if cell AP is not valid 
    if ((min(v) > -60) or (max(v) < 0)):
        return 50000000 

    # Voltage/APD features#######################
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    dvdt_max = np.max(np.diff(v[0:30])/np.diff(t[0:30]))

    ap_features['Vm_peak'] = max_p
    #ap_features['Vm_t'] = t[max_p_idx]
    ap_features['dvdt_max'] = dvdt_max

    for apd_pct in [40, 50, 90]:
        apd_val = calc_APD(t,v,apd_pct) 
        ap_features[f'apd{apd_pct}'] = apd_val
 
    ap_features['triangulation'] = ap_features['apd90'] - ap_features['apd40']
    ap_features['RMP'] = np.mean(v[len(v)-50:len(v)])

    # Calcium/CaT features######################## 
    max_cai = np.max(cai)
    max_cai_idx = np.argmax(cai)
    max_cai_time = t[max_cai_idx]
    cat_amp = np.max(cai) - np.min(cai)
    ap_features['cat_amp'] = cat_amp * 1e5 #added in multiplier since number is so small
    ap_features['cat_peak'] = max_cai_time

    for cat_pct in [90]:
        cat_recov = max_cai - cat_amp * cat_pct / 100
        idx_catd = np.argmin(np.abs(cai[max_cai_idx:] - cat_recov))
        catd_val = t[idx_catd+max_cai_idx]

        ap_features[f'cat{cat_pct}'] = catd_val 

    return ap_features

def get_normal_sim_dat(mod, proto):
    """
        Runs simulation for a given individual. If the individuals is None,
        then it will run the baseline model
        Returns
        ------
            t, v, cai, i_ion
    """
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    sim = myokit.Simulation(mod,proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats
    dat = sim.run(5000)
    IC = sim.state()

    # Get t, v, and cai for second to last AP#######################
    #t, v, cai, i_ion = get_last_ap(dat, -2)

    #return (t, v, cai, i_ion, IC)
    return(dat, IC)

def get_last_ap(dat, AP):

    # Get t, v, and cai for second to last AP#######################
    i_stim = dat['stimulus.i_stim']

    # This is here so that stim for EAD doesnt interfere with getting the whole AP
    for i in list(range(0,len(i_stim))):
        if abs(i_stim[i])<50:
            i_stim[i]=0

    peaks = find_peaks(-np.array(i_stim), distance=100)[0]
    start_ap = peaks[AP] #TODO change start_ap to be after stim, not during
    end_ap = peaks[AP+1]

    t = np.array(dat['engine.time'][start_ap:end_ap])
    t = t - t[0]
    max_idx = np.argmin(np.abs(t-995))
    t = t[0:max_idx]
    end_ap = start_ap + max_idx

    v = np.array(dat['membrane.v'][start_ap:end_ap])
    cai = np.array(dat['intracellular_ions.cai'][start_ap:end_ap])
    i_ion = np.array(dat['membrane.i_ion'][start_ap:end_ap])

    return (t, v, cai, i_ion)

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

def get_ead_error(ind, code): 

    ## EAD CHALLENGE: Istim = -.1
    if code == 'stim':
        mod, proto = get_ind_data(ind)
        proto.schedule(5.3, 0.1, 1, 1000, 0) 
        proto.schedule(0.1, 3004, 1000-100, 1000, 1) #EAD amp is about 4mV from this
        sim = myokit.Simulation(mod,proto)
        dat = sim.run(5000)

        # Get t, v, and cai for second to last AP#######################
        t, v, cai, i_ion = get_last_ap(dat, -2)

    ## EAD CHALLENGE: ICaL = 15x (acute increase - no prepacing here)
    if code == 'ical':
        mod, proto = get_ind_data(ind)
        mod['multipliers']['i_cal_pca_multiplier'].set_rhs(ind[0]['i_cal_pca_multiplier']*13)
        proto.schedule(5.3, 0.1, 1, 1000, 0) 
        sim = myokit.Simulation(mod,proto)
        dat = sim.run(5000)

        # Get t, v, and cai for second to last AP#######################
        t, v, cai, i_ion = get_last_ap(dat, -2)

    ## EAD CHALLENGE: IKr = 90% block (acute increase - no prepacing here)
    if code == 'ikr':
        mod, proto = get_ind_data(ind)
        mod['multipliers']['i_kr_multiplier'].set_rhs(ind[0]['i_kr_multiplier']*0.05)
        mod['multipliers']['i_kb_multiplier'].set_rhs(ind[0]['i_kb_multiplier']*0.05)
        proto.schedule(5.3, 0.1, 1, 1000, 0) 
        sim = myokit.Simulation(mod,proto)
        dat = sim.run(5000)

        # Get t, v, and cai for second to last AP#######################
        t, v, cai, i_ion = get_last_ap(dat, -2)


    ########### EAD DETECTION ############# 
    EAD = detect_EAD(t,v)


    return t,v,EAD

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

def detect_APD(t, v, apd90_base):
    APD90_i = calc_APD(t, v, 90)
    APD90_error = (APD90_i - apd90_base)/(APD90_i)*100
    if APD90_error < 40:
        result_APD = 0
    else:
        result_APD = 1
    return(result_APD)

def calc_APD(t, v, apd_pct):
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    repol_pot = max_p - apa * apd_pct/100
    idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
    apd_val = t[idx_apd+max_p_idx]
    return(apd_val) 

def rrc_search(IC, ind):
    #Run 6 normal beats and 1 at 0.3 stim to assess RRC 
    all_t = []
    all_v = []
    stims = [0, 0.3]

    mod, proto = get_ind_data(ind)
    mod.set_state(IC) #use state after prepacing
    proto.schedule(5.3, 0.2, 1, 1000, 0)
    proto.schedule(0.3, 5004, 995, 1000, 1)
    sim = myokit.Simulation(mod, proto)
    dat = sim.run(7000)

    t0, v0, cai0, i_ion0 = get_last_ap(dat, 4)
    all_t.append(t0)
    all_v.append(v0)
    result_EAD0 = detect_EAD(t0,v0)
    result_RF0 = detect_RF(t0,v0)

    t3, v3, cai3, i_ion3 = get_last_ap(dat, 5)
    all_t.append(t3)
    all_v.append(v3)
    result_EAD3 = detect_EAD(t3,v3)
    result_RF3 = detect_RF(t3,v3)

    if result_EAD0 == 1 or result_RF0 == 1:
        RRC = 0

    elif result_EAD3 == 0 and result_RF3 == 0:
        # no abnormality at 0.3 stim, return RRC
        RRC = 0.3

    else:
        low = 0.075
        high = 0.3
        while (high-low)>0.0025:
            mid = round((low + (high-low)/2), 4)
            stims.append(mid)

            mod, proto = get_ind_data(ind)
            mod.set_state(IC) #use state after prepacing
            proto.schedule(5.3, 0.2, 1, 1000, 0)
            proto.schedule(mid, 4004, 995, 1000, 1)
            sim = myokit.Simulation(mod, proto)
            dat = sim.run(6000)

            t, v, cai, i_ion = get_last_ap(dat, 4)
            all_t.append(t)
            all_v.append(v)
            result_EAD = detect_EAD(t,v)
            result_RF = detect_RF(t,v)

            if (high-low)<0.0025:
                break 
            
            elif result_EAD == 0 and result_RF == 0:
                # no RA so go from mid to high
                low = mid

            else:
                #repolarization failure so go from mid to low 
                high = mid

        result_EADlast = detect_EAD(all_t[-1],all_v[-1])
        result_RFlast = detect_RF(all_t[-1],all_v[-1])
        if result_EADlast == 0 and result_RFlast == 0:
            RRC = stims[-1] #the last stim attempted should have a repolarization abnormality
        else:
            RRC = stims[-2]

    return(RRC)


#%% SET THRESHOLD AND UPDATE LIST OF BEST INDIVIDUALS 
best_error1 = []
best_ind1 = []
challenges1 = []

best_error_ab = []
best_ind_ab = []
challenges_ab = []

for i in list(range(0,len(best_ind))):
    if challenges[i][0]==0 and challenges[i][1]==0 and challenges[i][2]==0 and challenges[i][3]==0:
        best_error1.append(best_error[i][0])
        best_ind1.append(best_ind[i]) 
        challenges1.append(challenges[i])
    else:
        best_error_ab.append(best_error[i][0])
        best_ind_ab.append(best_ind[i]) 
        challenges_ab.append(challenges[i])

print("ind1 length", len(best_ind1))

#%% explore what is different about the inds that were not immune to all challenges

conductance_groups_ab = []
error_groups_ab = []

for cond in list(range(0, len(best_ind_ab[0]))):
    current_cond_ab = []
    current_error_ab = []

    for gen in list(range(0, len(best_ind_ab))):
        current_cond_ab.append(best_ind_ab[gen][cond])
        current_error_ab.append(best_error_ab[gen])

    conductance_groups_ab.append(current_cond_ab) 
    error_groups_ab.append(current_error_ab)

for i in list(range(1,len(conductance_groups_ab)+1)):
    sc = plt.scatter([i]*len(conductance_groups_ab[0]), conductance_groups_ab[i-1])

positions = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
label = ('GCaL', 'GKs', 'GKr', 'GNaL', 'GNa', 'Gto', 'GK1', 'GNCX', 'GNaK', 'Gkb')
plt.ylabel("Conductance Value")
plt.xticks(positions, label)
plt.savefig(path + '\\inds_abnormal.png')
plt.show()

#%% PLOT & CALCULATE MEAN AND STANDARD DEVIATION
means = []
stds = []
conductance_groups = []
error_groups = []

for cond in list(range(0, len(best_ind1[0]))):
    current_cond = []
    current_error = []

    for gen in list(range(0, len(best_ind1))):
        current_cond.append(best_ind1[gen][cond])
        current_error.append(best_error1[gen])

    conductance_groups.append(current_cond) 
    error_groups.append(current_error)
    mean = np.mean(current_cond)
    means.append(mean)
    std = np.std(current_cond)
    stds.append(std)

for i in list(range(1,len(means)+1)):
    sc = plt.scatter([i]*len(conductance_groups[0]), conductance_groups[i-1])

positions = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
label = ('GCaL', 'GKs', 'GKr', 'GNaL', 'GNa', 'Gto', 'GK1', 'GNCX', 'GNaK', 'Gkb')
plt.ylabel("Conductance Value")
plt.xticks(positions, label)
plt.savefig(path + '\\best_all_gens.png')
plt.show()

#%% WRITE CSV
label = ['GCaL', 'GKs', 'GKr', 'GNaL', 'GNa', 'Gto', 'GK1', 'GNCX', 'GNaK', 'Gkb']
keys = [val for val in label]
dict_cond = dict(zip(keys, conductance_groups))
df_cond = pd.DataFrame(dict_cond)  
df_cond.to_csv(path + '\\FINAL_conds.csv', index=False)

dict_error = dict(zip(keys, error_groups))
df_error = pd.DataFrame(dict_error)  
df_error.to_csv(path + '\\FINAL_error.csv', index=False)

dict_mean = [dict(zip(keys, means))]
df_mean = pd.DataFrame(dict_mean)  
df_mean.to_csv(path + '\\means.csv', index=False)

dict_stds = [dict(zip(keys, stds))]
df_stds = pd.DataFrame(dict_stds)  
df_stds.to_csv(path + '\\stds.csv', index=False)

#%% CORRELATION ANALYSIS

corrMatrix = df_cond.corr()
print (corrMatrix)
sn.heatmap(corrMatrix, annot=True)
plt.savefig(path + '\\corr_matrix.png')
plt.show()

#%% ANALYSIS OF ALL TRIALS
frames = []
trials = ['trial1', 'trial2', 'trial3', 'trial4', 'trial6', 'trial7', 'trial8', 'trial9', 'trial10']
for i in list(range(0,len(trials))):
    path = 'c:\\Users\\Kristin\\Desktop\\iter4\\g100_p200_e2\\'+ trials[i]
    df = pd.read_csv(path + '\\FINAL_conds.csv')
    t = [i]*len(df)
    df['Trial'] = t
    frames.append(df)

all_conds = pd.concat(frames)

path_trials = 'c:\\Users\\Kristin\\Desktop\\iter4\\g100_p200_e2'
all_conds.to_csv(path_trials + '\\all_conds.csv', index=False)

colors = {0:'red', 1:'orange', 2:'yellow', 3:'green', 4:'blue', 5:'purple', 6:'cyan', 7:'grey', 8:'pink'}
c = ['GCaL', 'GKs', 'GKr', 'GNaL', 'GNa', 'Gto', 'GK1', 'GNCX', 'GNaK', 'Gkb']
for i in list(range(1,len(all_conds.columns))):
    cond = all_conds.columns[i-1]
    sc = plt.scatter([i]*len(all_conds[cond]), all_conds[cond], c=all_conds['Trial'].map(colors), alpha=0.01)

positions = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
label = ('GCaL', 'GKs', 'GKr', 'GNaL', 'GNa', 'Gto', 'GK1', 'GNCX', 'GNaK', 'Gkb')
plt.ylabel("Conductance Value")
plt.xticks(positions, label)
plt.savefig(path_trials + '\\best_all_trials.png')
plt.show()

corrMatrix = all_conds.corr()
print(corrMatrix)
sn.heatmap(corrMatrix, annot=True)
plt.savefig(path_trials + '\\corr_matrix_alltrials.png')
plt.show()

#%% ONCE FINAL GROUP IS CHOSEN, ASSESS DRUGS FROM PASSINI 2017 AND TOMEK 2019


# %%
