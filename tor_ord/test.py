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
import time

#%%
print(time.time())
mod, proto, x = myokit.load('./tor_ord_endo2.mmt')
## RRC CHALLENGE
stims = [0, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
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
dat = sim.run(66000)
print(time.time())

#%%

mod, proto, x = myokit.load('./tor_ord_endo2.mmt')
proto.schedule(5.3, 0.2, 1, 1000, 0)
#proto.schedule(0.2, 4004, 995, 1000, 1)

print(time.time())
sim = myokit.Simulation(mod, proto)
print(time.time())

dat = sim.run(6000)
sim.reset()
proto.schedule(.2, 5004, 995, 1000, 0)
sim.set_protocol(proto)
dat1 = sim.run(6000)

plt.plot(dat['engine.time'], dat['membrane.v'], label = 'before change')
plt.plot(dat1['engine.time'], dat1['membrane.v'], label = 'after change')
plt.legend()


#%%
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


mod, proto, x = myokit.load('./tor_ord_endo2.mmt')
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
opt = [1.2216093580100347, 0.401069084533474, 1.0692700180573773, 0.7121264982301789, 1.5499061545040453, 1.2461438821369062, 1.1630493338301395, 0.665419266548533, 0.6943830525966528, 1.4710083729878485]
keys = [val for val in tunable_parameters]
optimized = [dict(zip(keys[0], opt))]

for k, v in optimized[0].items():
    mod['multipliers'][k].set_rhs(v)

print(time.time())
#stims = [0, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
stims = np.linspace(0.1, 0.125, 11)
AP = [4, 5004, 10004, 15004, 20004, 25004, 30004, 35004, 40004, 45004, 50004]

proto.schedule(5.3, 0.2, 1, 1000, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(100*1000) 
IC = sim.state()

all_t = []
all_v = []

for i in list(range(0,len(stims))):
    sim.reset()
    sim.set_state(IC)
    proto.schedule(stims[i], AP[i], 995, 1000, 1)
    sim.set_protocol(proto)
    dat = sim.run(AP[i]+2000)
    t, v, cai, i_ion = get_last_ap(dat, int((AP[i]-4)/1000))
    all_t.append(t)
    all_v.append(v)
print(time.time())

plt.figure()
plt.plot(dat['engine.time'], dat['membrane.v'])

plt.figure()
plt.figure(figsize=[20,5])
for i in list(range(0, len(all_t))):
    plt.plot(all_t[i], all_v[i], label = "stim = " + str(-stims[i]) +" A/F")

#%


#%%
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

print(time.time())
stims = [0, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
AP = [4, 5004, 10004, 15004, 20004, 25004, 30004, 35004, 40004, 45004, 50004]
mod, proto, x = myokit.load('./tor_ord_endo2.mmt')
proto.schedule(5.3, 0.2, 1, 1000, 0)
sim = myokit.Simulation(mod, proto)
for i in list(range(0,len(stims))):
    sim.reset()
    proto.schedule(stims[i], AP[i], 995, 1000, 1)
    sim.set_protocol(proto)
    dat = sim.run(AP[i]+2000)
print(time.time())


#%%
# READ IN DATA 
path = 'c:\\Users\\Kristin\\Desktop\\iter4\\search\\trial2'
gen = 99
gen_name = 'gen99'

pop = pd.read_csv(path + '\\pop.csv')
error = pd.read_csv(path + '\\error.csv')

last_gen = []
for i in list(range(0, len(pop[gen_name]))):
    last_gen.append(literal_eval(pop[gen_name][i]))

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

def get_rrc_error(mod, proto, IC):

    ## RRC CHALLENGE
    stims = [0, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
    #stims = [0, 0.075, 0.15, 0.2, 0.25, 0.3]
    #stims = [0, 0.075, 0.1, 0.125, 0.15, 0.175]

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

    #################### RRC DETECTION & ERROR CALCULATION ###########################

    #pos_error = [2500, 2000, 1500, 1000, 500, 0]
    pos_error = [5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 1000, 500, 0]
    for v in list(range(1, len(vals))): 
        if vals[v] == 1:
            RRC = -stims[v-1] #RRC will be the value before the first RF or EAD
            E_RRC = pos_error[v-1]
            break
        else:
            RRC = -stims[-1] #if there is no EAD or RF or APD>40% than the stim was not strong enough so error should be zero
            E_RRC = 0

    return dat, all_t, all_v, RRC, E_RRC

def get_rrc_error2(mod, proto, IC, RRC, E_RRC, cost):

    #################### RRC DETECTION & ERROR CALCULATION ##########################
    error = 0

    if cost == 'function_1':
        
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

        error += round((0.3 - (np.abs(RRC1)))*20000)

    else:
        # This just returns the error from the first RRC protocol
        error += E_RRC

    return error, RRC1, all_t, all_v

#%% 
best_index = np.where(error[gen_name]==min(error[gen_name]))[0][0]
tunable_parameters=['i_cal_pca_multiplier', 'i_ks_multiplier', 'i_kr_multiplier', 'i_nal_multiplier', 'i_na_multiplier', 'i_to_multiplier', 'i_k1_multiplier', 'i_NCX_multiplier', 'i_nak_multiplier', 'i_kb_multiplier']
base = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

keys = [val for val in tunable_parameters]
baseline = [dict(zip(keys, base))]
optimized = [dict(zip(keys, last_gen[best_index]))]

stims = [0, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3]
# Tor-ord Baseline
print(time.time())
m, p = get_ind_data(baseline)
t, v, cai, i_ion, IC = get_normal_sim_dat(m, p)
mod, proto = get_ind_data(baseline)
dat, t_rrc, v_rrc, rrc, E_RRC  = get_rrc_error(mod, proto, IC)
print(time.time())
print(rrc)
print(E_RRC)

# Optimized Model
print(time.time())
m1, p1 = get_ind_data(optimized)
t1, v1, cai1, i_ion1, IC1 = get_normal_sim_dat(m1, p1) 
mod1, proto1 = get_ind_data(optimized)
dat1, t_rrc1, v_rrc1, rrc1, E_RRC1 = get_rrc_error(mod1, proto1, IC1)
print(time.time())
print(rrc1)
print(E_RRC1)

#%% GA4
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

    return(RRC, all_t, all_v, stims)

def get_rrc_error(RRC, cost):

    #################### RRC DETECTION & ERROR CALCULATION ##########################
    error = 0
    RRC_est = RRC

    if cost == 'function_1':
        error += round((0.3 - (np.abs(RRC)))*20000)

    else:
        # This just returns the error from the first RRC protocol
        stims = np.asarray([0, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3])
        pos_error = [5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 1000, 500, 0]
        i = (np.abs(stims - RRC)).argmin()
        check_low = stims[i]-stims[i-1]
        check_high = stims[i]-stims[i+1]

        if check_high<check_low:
            RRC_est = stims[i-1]
            error = pos_error[i-1]
        else:
            RRC_est = i
            error += pos_error[i]

    return error, RRC_est


#%% RRC Calculation - baseline
print(time.time())
RRC, all_t, all_v, stims = rrc_search(IC, baseline)
error, rrc_est = get_rrc_error(RRC, 'function_1')
print(time.time())
print(RRC, error) 

# RRC Calculation - immunized
print(time.time())
RRC1, all_t1, all_v1, stims1 = rrc_search(IC1, optimized)
error1, rrc_est1 = get_rrc_error(RRC1, 'function_1')
print(time.time())
print(RRC1, error1) 
# %%
