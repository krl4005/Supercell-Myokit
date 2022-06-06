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
    t, v, cai, i_ion = get_last_ap(dat, -2)

    return (t, v, cai, i_ion, IC)

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

def get_rrc_error1(mod, proto, IC):
    ## RRC CHALLENGE
    stims = [0, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
    #stims = [0, 0.075, 0.1, 0.125, 0.15, 0.175]
    #stims = [0, 0.075, 0.15, 0.2, 0.25, 0.3]

    mod.set_state(IC) #use state after prepacing
    proto.schedule(5.3, 0.2, 1, 1000, 0)
    proto.schedule(stims[0], 4, 995, 1000, 1)
    proto.schedule(stims[1], 5004, 995, 1000, 1)
    proto.schedule(stims[2], 10004, 995, 1000, 1)
    proto.schedule(stims[3], 15004, 995, 1000, 1)
    proto.schedule(stims[4], 20004, 995, 1000, 1)
    proto.schedule(stims[5], 25004, 995, 1000, 1)
    proto.schedule(stims[6], 30004, 995, 1000, 1)
    proto.schedule(stims[7], 35004, 995, 1000, 1)
    proto.schedule(stims[8], 40004, 995, 1000, 1)
    proto.schedule(stims[9], 45004, 995, 1000, 1)
    proto.schedule(stims[10], 50004, 995, 1000, 1)

    sim = myokit.Simulation(mod, proto)
    dat = sim.run(52000)
    #dat = sim.run(28000)

    #t_base, v_base, cai_base, i_ion_base = t, v, cai, i_ion = get_last_ap(dat, 0)
    #apd90_base = detect_APD(t_base, v_base, 90)

    # Pull out APs with RRC stimulus 
    vals = []

    for i in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        #for i in [0, 5, 10, 15, 20, 25]:
        t, v, cai, i_ion = get_last_ap(dat, i)
        #plt.plot(t, v)

        ########### EAD DETECTION ############# 
        result_EAD = detect_EAD(t,v) 

        ########### RF DETECTION ############# 
        result_RF = detect_RF(t,v)

        ########### APD90 DETECTION ############
        #result_APD = detect_APD(t, v, apd90_base)

        # if EAD and RF place 0 in val list 
        # 0 indicates no RF or EAD for that RRC challenge
        if result_EAD == 0 and result_RF == 0: #and result_APD == 0:
            vals.append(0)
        else:
            vals.append(1)
        
        pos_error = [5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 1000, 500, 0]
        #pos_error = [2500, 2000, 1500, 1000, 500, 0]
        for v in list(range(1, len(vals))): 
            if vals[v] == 1:
                RRC = -stims[v-1] #RRC will be the value before the first RF or EAD
                E_RRC = pos_error[v-1]
                break
            else:
                RRC = -stims[-1] #if there is no EAD or RF or APD>40% than the stim was not strong enough so error should be zero
                E_RRC = 0

    return RRC, E_RRC

def get_rrc_error2(mod, proto, IC, RRC, E_RRC, cost):
    ## RRC CHALLENGE
    
    stims = [0, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
    start = np.where(stims == np.abs(RRC))[0][0]
    stims_narrow = np.linspace(stims[start], stims[start+1], 11)

    mod.set_state(IC) #start from state after first RRC protocol
    proto.schedule(5.3, 0.2, 1, 1000, 0)
    proto.schedule(stims_narrow[0], 4, 995, 1000, 1)
    proto.schedule(stims_narrow[1], 5004, 995, 1000, 1)
    proto.schedule(stims_narrow[2], 10004, 995, 1000, 1)
    proto.schedule(stims_narrow[3], 15004, 995, 1000, 1)
    proto.schedule(stims_narrow[4], 20004, 995, 1000, 1)
    proto.schedule(stims_narrow[5], 25004, 995, 1000, 1)
    proto.schedule(stims_narrow[6], 30004, 995, 1000, 1)
    proto.schedule(stims_narrow[7], 35004, 995, 1000, 1)
    proto.schedule(stims_narrow[8], 40004, 995, 1000, 1)
    proto.schedule(stims_narrow[9], 45004, 995, 1000, 1)
    proto.schedule(stims_narrow[10], 50004, 995, 1000, 1)

    sim = myokit.Simulation(mod, proto)
    dat = sim.run(52000)

    # Pull out APs with RRC stimulus 
    vals = []
    all_t = []
    all_v = []

    for i in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        #for i in [0, 5, 10, 15, 20, 25]:
        t, v, cai, i_ion = get_last_ap(dat, i)
        all_t.append(t)
        all_v.append(v)
        #plt.plot(t, v)

        ########### EAD DETECTION ############# 
        result_EAD = detect_EAD(t,v) 

        ########### RF DETECTION ############# 
        result_RF = detect_RF(t,v)

        ########### APD90 DETECTION ############
        #result_APD = detect_APD(t, v, apd90_base)

        # if EAD and RF place 0 in val list 
        # 0 indicates no RF or EAD for that RRC challenge
        if result_EAD == 0 and result_RF == 0: #and result_APD == 0:
            vals.append(0)
        else:
            vals.append(1)
        
        for v in list(range(1, len(vals))): 
            if vals[v] == 1:
                RRC1 = -stims_narrow[v-1] #RRC will be the value before the first RF or EAD
                break
            else:
                RRC1 = -stims_narrow[-1] #if there is no EAD or RF than the stim was not strong enough so error should be the same

    #################### RRC DETECTION & ERROR CALCULATION ##########################
    error = 0

    if cost == 'function_1':

        error += (0.3 - (np.abs(RRC1)))*10000
    else:
        # This just returns the error from the first RRC protocol
        error += E_RRC

    return error, RRC1, all_t, all_v

# Setting parameters
tunable_parameters=['i_cal_pca_multiplier',
                    'i_ks_multiplier',
                    'i_kr_multiplier',
                    'i_nal_multiplier',
                    'i_na_multiplier',
                    'i_to_multiplier',
                    'i_k1_multiplier',
                    'i_NCX_multiplier',
                    'i_nak_multiplier',
                    'i_kb_multiplier'],

base = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

keys = [val for val in tunable_parameters]
baseline = [dict(zip(keys[0], base))]
#%%
m, p = get_ind_data(baseline)
t, v, cai, i_ion, IC = get_normal_sim_dat(m, p)

mod, proto = get_ind_data(baseline)
rrc, E_RRC, IC_opt = get_rrc_error1(mod, proto, IC)

mod1, proto1 = get_ind_data(baseline)
error_opt, rrc_opt, t_opt, v_opt = get_rrc_error2(mod1, proto1, IC, rrc, E_RRC, 'function_1')

stims = [0, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
start = np.where(stims == np.abs(rrc))[0][0]
stims_narrow = np.linspace(stims[start], stims[start+1], 11)

print(rrc)
print(rrc_opt)
print(E_RRC)
print(error_opt)

plt.figure(figsize=[20,5])
for i in list(range(0, len(v_opt))):
    plt.plot(t_opt[i], v_opt[i], label = "stimulus = " + str(-stims_narrow[i])+ " A/F")
    
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()

# %%
