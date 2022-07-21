#%% Import functions and tools
from math import log10
import random
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from scipy.signal import find_peaks # pip install scipy
from multiprocessing import Pool


# %% Define Functions

def get_ind_data(ind):
    mod, proto, x = myokit.load('./tor_ord_endo2.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto

def immunize_ind_data(ind):
    immunization_profile = [0.3, 1, 2, 1.7, 0.6, 1.6, 1.9, 0.1, 1.5, 1.9]
    vals = list(ind.values())
    labs = list(ind.keys())
    new_conds = [immunization_profile[i]*vals[i] for i in range(len(immunization_profile))]
    immune_ind = dict(zip(labs, new_conds))

    return immune_ind

def initialize_individuals():
    """
    Creates the initial population of individuals. The initial 
    population 
    Returns:
        An Individual with conductance parameters 
    """
    tunable_parameters=['i_cal_pca_multiplier','i_ks_multiplier','i_kr_multiplier','i_nal_multiplier','i_na_multiplier','i_to_multiplier','i_k1_multiplier','i_NCX_multiplier','i_nak_multiplier','i_kb_multiplier']
    
    # Builds a list of parameters using random upper and lower bounds.
    lower_exp = log10(0.1)
    upper_exp = log10(2)

    initial_params = []
    for i in range(0, len(tunable_parameters)):
            initial_params.append(10**random.uniform(lower_exp, upper_exp))

    keys = [val for val in tunable_parameters]
    return dict(zip(keys, initial_params))

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

    #convert to list for accurate storage in dataframe
    t = list(t)
    v = list(v)

    return (t, v)

def visualize_raw_pop(ind): 
    mod, proto, x = myokit.load('./tor_ord_endo2.mmt')
    if ind is not None:
        for k, v in ind.items():
            mod['multipliers'][k].set_rhs(v)

    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    sim = myokit.Simulation(mod,proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats
    dat = sim.run(5000) 

    #t = dat['engine.time']
    #v = dat['membrane.v']
    #return(t,v)
    return(dat)

def assess_challenges(ind):

    ## EAD CHALLENGE: Istim = -.1
    mod, proto = get_ind_data(ind)
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    proto.schedule(0.1, 2004, 1000-100, 1000, 1) #EAD amp is about 4mV from this
    sim = myokit.Simulation(mod,proto)
    sim.pre(100*1000)
    dat = sim.run(4000) #pre-pace for 100 beats
    IC = sim.state()

    ## EAD CHALLENGE: ICaL = 30x (acute increase - no prepacing here)
    #sim.reset()
    sim.set_state(IC)
    sim.set_constant('multipliers.i_cal_pca_multiplier', ind[0]['i_cal_pca_multiplier']*30)
    dat1 = sim.run(1000)

    ## EAD CHALLENGE: IKr = 80% block (acute increase - no prepacing here)
    #sim.reset()
    sim.set_state(IC)
    sim.set_constant('multipliers.i_cal_pca_multiplier', ind[0]['i_cal_pca_multiplier'])
    sim.set_constant('multipliers.i_kr_multiplier', ind[0]['i_kr_multiplier']*0.05)
    sim.set_constant('multipliers.i_kb_multiplier', ind[0]['i_kb_multiplier']*0.05)
    dat2 = sim.run(1000)

    # Get Specific APs
    t_base, v_base = get_last_ap(dat, 1)
    t_ead, v_ead = get_last_ap(dat, -2)
    t_ical = list(dat1['engine.time'])
    v_ical = list(dat1['membrane.v'])
    t_rf = list(dat2['engine.time'])
    v_rf = list(dat2['membrane.v'])

    return t_base, v_base, t_ead, v_ead, t_ical, v_ical, t_rf, v_rf

# %% Generate baseline and immunized populations - TO RUN ON CLUSTER


time1 = time.time()

def collect_data(i):
    print(i)
    ind = [initialize_individuals()]
    ind_imm = [immunize_ind_data(ind[0])]
    t_base, v_base, t_ead, v_ead, t_ical, v_ical, t_rf, v_rf = assess_challenges(ind)
    t_base_imm, v_base_imm, t_ead_imm, v_ead_imm, t_ical_imm, v_ical_imm, t_rf_imm, v_rf_imm = assess_challenges(ind_imm)

    labels = ['ind', 't', 'v', 't_ead', 'v_ead', 't_ical', 'v_ical', 't_rf', 'v_rf', 'ind_imm', 't_imm', 'v_imm', 't_ead_imm', 'v_ead_imm', 't_ical_imm', 'v_ical_imm', 't_rf_imm', 'v_rf_imm' ]
    vals = [ind, t_base, v_base, t_ead, v_ead, t_ical, v_ical, t_rf, v_rf, ind_imm, t_base_imm, v_base_imm, t_ead_imm, v_ead_imm, t_ical_imm, v_ical_imm, t_rf_imm, v_rf_imm]

    data = dict(zip(labels, vals))
    return(data)

if __name__ == "__main__":
    #num_models = 50000
    num_models = 10
    p = Pool() #allocates for the maximum amount of processers on our screen 
    #result = p.map(assess_challenges, models)
    #result_immune = p.map(assess_challenges, immune_models)
    result = p.map(collect_data, range(num_models))
    p.close()
    p.join()

time2 = time.time()
print('processing time: ', (time2-time1)/60, ' Minutes')

# %% Generate population and store data - TO RUN LOCALLY
"""
num_models = 50

models = [[initialize_individuals()] for i in list(range(0, num_models))]
immune_models = [[immunize_ind_data(ind[0])] for ind in models] 

time1 = time.time()

result = [assess_challenges(m) for m in models]
result_immune = [assess_challenges(i) for i in immune_models]

time2 = time.time()
print('processing time: ', (time2-time1)/60, ' Minutes')

"""

#%% Save Data

labels = ['t', 'v', 't_ead', 'v_ead', 't_ical', 'v_ical', 't_rf', 'v_rf']
#df_data = pd.DataFrame(result, columns=labels)
df_data = pd.DataFrame(result)
df_data.to_csv("data.csv")

#df_imm_data = pd.DataFrame(result_immune, columns=labels)
#df_imm_data.to_csv("immune_data.csv")

#df_models = pd.DataFrame(models)
#df_models.to_csv("models.csv")

#df_imm_models = pd.DataFrame(immune_models)
#df_imm_models.to_csv("immune_models.csv")

