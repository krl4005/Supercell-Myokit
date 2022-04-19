#%% Run Simulation and Plot AP
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 
from scipy.signal import find_peaks # pip install scipy

# %% FUNCTIONS
def get_RF_error(t,v):

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
        result = "Repolarization failure!"
    else:
        result = "normal repolarization - resting membrane potential from t=", no_RF[0][0], "to t=", no_RF[0][len(no_RF[0])-1]

    return result

def get_last_ap(dat):
    
    i_stim = dat['stimulus.i_stim']

    # This is here so that stim for EAD doesnt interfere with getting the whole AP
    for i in list(range(0,len(i_stim))):
        if abs(i_stim[i])<50:
            i_stim[i]=0

    peaks = find_peaks(-np.array(i_stim), distance=100)[0]
    start_ap = peaks[-2] #TODO change start_ap to be after stim, not during
    end_ap = peaks[-1]

    #print(start_ap, dat['engine.time'][start_ap])
    #print(end_ap, dat['engine.time'][end_ap])

    t = np.array(dat['engine.time'][start_ap:end_ap])
    t = t - t[0]
    max_idx = np.argmin(np.abs(t-995))
    t = t[0:max_idx]
    end_ap = start_ap + max_idx

    v = np.array(dat['membrane.v'][start_ap:end_ap])
    cai = np.array(dat['intracellular_ions.cai'][start_ap:end_ap])
    i_ion = np.array(dat['membrane.i_ion'][start_ap:end_ap])

    return (t, v, cai, i_ion)

def get_ead_error(mod, proto, sim, ind): 
    
    ## EAD CHALLENGE: ICaL = 15x (acute increase - no prepacing here)
    #mod['multipliers']['i_cal_pca_multiplier'].set_rhs(ind[0]['i_cal_pca_multiplier']*15)

    ## EAD CHALLENGE: Istim = -.1
    proto.schedule(0.1, 3004, 1000-100, 1000, 1)
    sim = myokit.Simulation(mod, proto)
    dat = sim.run(5000)

    ########### EAD DETECTION ############# 
    t,v,cai,i_ion = get_last_ap(dat)
    #plt.plot(t, v)

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
        result = "no EAD"
    else:
        result = "EAD:", round(max(EADs))

    #################### ERROR CALCULATION #######################
    #error = 0

    #if GA_CONFIG.cost == 'function_1':
    #    error += (0 - (10*EAD))**2
    #else:
    #    error += 10*EAD

    #return error
    return t, v, result

def get_ind_data(ind):
    mod, proto, x = myokit.load('./tor_ord_endo.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    proto.schedule(5.3, 0.1, 1, 1000, 0) #ADDED IN
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats 
    #IC = sim.state()

    return mod, proto, sim

def get_normal_sim_dat(sim):

    dat = sim.run(5000)

    # Get t, v, and cai for second to last AP#######################
    t, v, cai, i_ion = get_last_ap(dat)

    return (t, v, cai, i_ion)

#%% Set model
tunable_parameters=['i_cal_pca_multiplier',
                    'i_ks_multiplier',
                    'i_kr_multiplier',
                    'i_nal_multiplier',
                    'jup_multiplier'],
initial_params = [1, 1, 1, 1, 1]

keys = [val for val in tunable_parameters]
final = [dict(zip(keys[0], initial_params))]
print(final[0])

# Visualize AP
mod, proto, sim = get_ind_data(final)
t, v, cai, i_ion = get_normal_sim_dat(sim)
plt.plot(t, v, label = "normal")

# Calculate RF error
RF_result = get_RF_error(t,v)
print(RF_result)

# Calculate EAD error
t_EAD, v_EAD, result_EAD = get_ead_error(mod, proto, sim, final)
plt.plot(t_EAD, v_EAD, label = "ead")
print(result_EAD)

plt.legend()

# %%
