#%% Import functions and tools
from math import log10
import random
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 


# %% Define Functions
def split_APs(dat):
    v = dat['membrane.V']
    t = dat['engine.time']
    i_stim=np.array(dat['stimulus.i_stim'].tolist())
    AP_S_where = np.where(i_stim == -3.0)[0]
    AP_S_diff = np.where(np.diff(AP_S_where)!=1)[0]+1
    peaks = AP_S_where[AP_S_diff]

    v_APs = np.split(v, peaks)
    t_APs = np.split(t, peaks)
    s = np.split(i_stim, peaks)
    return(v_APs, t_APs)

def calc_APD90(v,t):
        mdp = min(v)
        max_p = max(v)
        max_p_idx = np.argmax(v)
        apa = max_p - mdp
        dvdt_max = np.max(np.diff(v[0:30])/np.diff(t[0:30]))

        repol_pot = max_p - apa * 90/100
        idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
        apd_val = t[idx_apd+max_p_idx]
        return apd_val

def initialize_individuals(lower_bound, upper_bound):

    tunable_parameters=['i_cal_pca_multiplier',
                    'i_ks_multiplier',
                    'i_kr_multiplier',
                    'i_na_multiplier',
                    'i_to_multiplier',
                    'i_k1_multiplier',
                    'i_f_multiplier']


    # Builds a list of parameters using random upper and lower bounds.
    lower_exp = log10(lower_bound)
    upper_exp = log10(upper_bound)
    
    initial_params = [] 
    for i in range(0, len(tunable_parameters)):
        initial_params = [10**random.uniform(lower_exp, upper_exp) for i in range(0, len(tunable_parameters))]

    keys = [val for val in tunable_parameters]

    return dict(zip(keys, initial_params)), initial_params

def visualize_raw_pop(ind): 
    mod, proto, x = myokit.load('./kernik.mmt')
    if ind is not None:
        for k, v in ind.items():
            mod['multipliers'][k].set_rhs(v)

    proto.schedule(0, 100, 5, 1000, 0) 
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats
    dat = sim.run(1000) 

    t = dat['engine.time']
    v = dat['membrane.V']
    plt.plot(t,v)

def filter_1(ind):
    mod, proto, x = myokit.load('./kernik.mmt')
    if ind is not None:
        for k, v in ind.items():
            mod['multipliers'][k].set_rhs(v)

    proto.schedule(0, 100, 5, 1000, 0) 
    mod['ik1']['g_K1'].set_rhs(mod['ik1']['g_K1'].value()*(11.24/5.67))
    mod['ina']['g_Na'].set_rhs(mod['ina']['g_Na'].value()*(187/129))
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats
    dat = sim.run(1000) 

    t = dat['engine.time']
    v = dat['membrane.V']

    # FILTER
    if np.max(v) < -70:
        plt.plot(t,v)
        model_conduct = ind
    else:
        model_conduct = [0]
        print("model: ", i)

    return model_conduct

def closest(lst, K):
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return idx   

def filter_2(ind):
    mod, proto, x = myokit.load('./kernik.mmt')
    if ind is not None:
        for k, v in ind.items():
            mod['multipliers'][k].set_rhs(v)

    proto.schedule(1, 100, 5, 1000, 0) 
    mod['ik1']['g_K1'].set_rhs(mod['ik1']['g_K1'].value()*(11.24/5.67))
    mod['ina']['g_Na'].set_rhs(mod['ina']['g_Na'].value()*(187/129))
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats
    dat = sim.run(1000) 

    t = dat['engine.time']
    v = dat['membrane.V']

    # FILTER
    end = closest(t,200)
    if np.max(v[0:end]) > 20:
        plt.plot(t,v)
        final_conduct = ind
    else:
        final_conduct = [0]
        t = [0]
        v = [0]
        print("model: ", i)

    return t,v, final_conduct

def get_rrc_error(mod, proto, sim):

    ## RRC CHALLENGE
    proto.schedule(1, 100.2, 5, 1000, 0)
    proto.schedule(0.1, 125, 500, 1000, 1)
    proto.schedule(0.15, 5125, 500, 1000, 1)
    proto.schedule(0.2, 10125, 500, 1000, 1)
    proto.schedule(0.25, 15125, 500, 1000, 1)
    proto.schedule(0.3, 20125, 500, 1000, 1)

    sim = myokit.Simulation(mod, proto)
    sim.pre(100 * 1000) #pre-pace for 100 beats
    dat = sim.run(25000)

    #plt.plot(dat['engine.time'], dat['stimulus.i_stim'], color = 'black')

    # Pull out APs and find APD90
    v_APs, t_APs = split_APs(dat)
    apd90s = []
    norm_APs = []
    for i in list(range(0, len(v_APs))):
        vol = v_APs[i]
        tim = t_APs[i] -1000*i
        apd90 = calc_APD90(vol, tim)
        apd90s.append(apd90)
        norm_APs.append(apd90)

    ########### APD90 DETECTION ############# 
    stim_idx = [0, 5, 10, 15, 20]
    stim_APs = []
    for i in stim_idx:
        stim_APs.append(apd90s[i])
        norm_APs.remove(apd90s[i])
    
    base_apd90 = np.mean(norm_APs)
    apd_change = []
    for i in list(range(0,len(stim_APs))):
        del_apd = ((stim_APs[i]-base_apd90)/base_apd90)*100
        apd_change.append(del_apd)
            
    #################### RRC DETECTION ###########################
    #global RRC

    RRC_vals = [-0.3, -0.45, -0.6, -0.5, -0.75, -0.9]
    
    for i in list(range(0, len(apd_change))): 
        if apd_change[i] > 40:
            RRC = RRC_vals[i]
            break
        else:
            RRC = -1.0 #if there is no EAD than the stim was not strong enough so error should be zero

    return dat, RRC, apd_change

def get_ind_data(ind):
    mod, proto, x = myokit.load('./kernik.mmt')
    if ind is not None:
        for k, v in ind.items():
            mod['multipliers'][k].set_rhs(v)

    proto.schedule(1, 100.1, 1, 1000, 0) #ADDED IN
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats 
    IC = sim.state()

    return mod, proto, sim, IC

# %% Generate population and visualize 
num_models = 50
models = []
models_dict = []
for i in list(range(0,num_models)):
    mod_dict, mod_ind = initialize_individuals(0.2, 2)
    visualize_raw_pop(mod_dict)
    models.append(mod_ind)
    models_dict.append(mod_dict)

#%% penalize against cells that are still spontaneous after IK1 dynamic clamp 
filter1_models = []
for i in list(range(0,len(models_dict))):
    cond = filter_1(models_dict[i])
    filter1_models.append(cond)
filter1_models = [i for i in filter1_models if i != [0]]

#%% penalize against all cells that do not look normal when pacing with dynamic clamp

final_models = []
v_all = []
t_all = []
for i in list(range(0,len(filter1_models))):
    t,v,conduct = filter_2(filter1_models[i])
    v_all.append(v)
    t_all.append(t)
    final_models.append(conduct)

final_models = [i for i in final_models if i != [0]]
v_all = [i for i in v_all if i != [0]]
t_all = [i for i in t_all if i != [0]]

#%% Calculate RRC for each model
all_RRCs = []
for i in list(range(0,len(final_models))):
    mod, proto, sim, IC = get_ind_data(final_models[i])
    dat, RRC, apd_change = get_rrc_error(mod, proto, sim)
    all_RRCs.append(RRC)

#%% plot high AR group
for i in list(range(0, len(all_RRCs))):
    if all_RRCs[i] > -0.5:
        #plt.plot(t_all[i], v_all[i], color = 'red', alpha=0.1*i)
        plt.plot(t_all[i], v_all[i], color = 'red', alpha = 0.5)
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")

# plot low AR group
for i in list(range(0, len(all_RRCs))):
    if all_RRCs[i] < -0.5:
        #plt.plot(t_all[i], v_all[i], color = 'blue', alpha=0.05*i)
        plt.plot(t_all[i], v_all[i], color = 'blue', alpha = 0.5)
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.legend()

# %% Calculate Variance
vari = []
for j in list(range(0, len(models[0]))):
    con = []

    for i in list(range(0,len(models))):
        con.append(models[i][j])

    vari.append(np.var(con))

print(vari)

#%%
plt.plot(dat['engine.time'], dat['membrane.V'], color = 'blue', label = 'model 1 APD > 40% increase when RR = -0.75')
plt.plot(t_all[i], v_all[i], color='blue', alpha = 0.3, label = 'model 1')
plt.xlim([-100,1000])
plt.legend()
plt.label(bbox_to_anchor = [0.5, 0.2])
