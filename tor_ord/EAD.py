#%% Run Simulation and Plot AP
from charset_normalizer import detect
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 
from scipy.signal import find_peaks # pip install scipy

# %% FUNCTIONS
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

def get_last_ap(dat, AP):
    
    i_stim = dat['stimulus.i_stim']

    # This is here so that stim for EAD doesnt interfere with getting the whole AP
    for i in list(range(0,len(i_stim))):
        if abs(i_stim[i])<50:
            i_stim[i]=0

    peaks = find_peaks(-np.array(i_stim), distance=100)[0]
    start_ap = peaks[AP] #TODO change start_ap to be after stim, not during
    end_ap = peaks[AP+1]

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

    t,v,cai,i_ion = get_last_ap(dat, -2)
    #plt.plot(t, v)

    ########### EAD DETECTION ############# 
    result = detect_EAD(t,v)

    #################### ERROR CALCULATION #######################
    #error = 0

    #if GA_CONFIG.cost == 'function_1':
    #    error += (0 - (10*EAD))**2
    #else:
    #    error += 10*EAD

    #return error
    return t, v, result

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
    #sim.pre(1000 * 100) #pre-pace for 100 beats
    dat = sim.run(5000)

    # Get t, v, and cai for second to last AP#######################
    t, v, cai, i_ion = get_last_ap(dat, -2)

    return (t, v, cai, i_ion)

def get_rrc_error(mod, proto, sim):

    ## RRC CHALLENGE
    stims = [0, 0.025, 0.05, 0.075, 0.1, 0.125]
    proto.schedule(5.3, 0.2, 1, 1000, 0)
    proto.schedule(stims[0], 4, 995, 1000, 1)
    proto.schedule(stims[1], 5004, 995, 1000, 1)
    proto.schedule(stims[2], 10004, 995, 1000, 1)
    proto.schedule(stims[3], 15004, 995, 1000, 1)
    proto.schedule(stims[4], 20004, 995, 1000, 1)
    proto.schedule(stims[5], 25004, 995, 1000, 1)

    sim = myokit.Simulation(mod, proto)
    sim.pre(100 * 1000) #pre-pace for 100 beats
    dat = sim.run(25000)

    # Pull out APs with RRC stimulus 
    vals = []
    for i in [0, 5, 10, 15, 20]:
        t, v, cai, i_ion = get_last_ap(dat, i)
        plt.plot(t, v)

        # Detect EAD
        result_EAD = detect_EAD(t,v) 

        # Detect RF
        result_RF = detect_RF(t,v)

        # if EAD and RF place 0 in val list 
        # 0 indicates no RF or EAD for that RRC challenge
        if result_EAD == 0 and result_RF == 0:
            vals.append(0)
        else:
            vals.append(1)

    #find first y in list --> that will represent the RRC
    pos_error = [5000, 4000, 3000, 2000, 1000, 0]
    for v in list(range(0, len(vals))): 
        if vals[v] == 1:
            RRC = -stims[v-1] #RRC will be the value before the first RF or EAD
            error = pos_error[v-1]
            break
        else:
            RRC = -1.25 #if there is no EAD or RF than the stim was not strong enough so error should be zero
            error = 0
    
    return vals, dat, RRC, error

#%% Set model
tunable_parameters=['i_cal_pca_multiplier',
                    'i_ks_multiplier',
                    'i_kr_multiplier',
                    'i_nal_multiplier',
                    'jup_multiplier'],
initial_params = [1, 1, 0.05, 1, 1]

keys = [val for val in tunable_parameters]
final = [dict(zip(keys[0], initial_params))]
print(final[0])


mod, proto, sim = get_ind_data(final)

# Calculate RRC 
plt.figure(1)
vals, dat, RRC, error = get_rrc_error(mod,proto,sim)
plt.figure(2)
plt.plot(dat['engine.time'], dat['membrane.v'])
print("0: no RF or EAD, 1: RF and/or EAD ", vals)
print("RRC: ", RRC)
print("error: ", error)

# Visualize AP
t, v, cai, i_ion = get_normal_sim_dat(sim)
plt.figure(3)
plt.plot(t, v, label = "normal")

# Calculate RF error
RF_result = detect_RF(t,v)
print("RF result: (0- no RF, 1- RF)", RF_result)

# Calculate EAD error
t_EAD, v_EAD, result_EAD = get_ead_error(mod, proto, sim, final)
plt.plot(t_EAD, v_EAD, label = "EAD Challenge")
print("EAD result: (0- no EAD, 1- EAD) ", result_EAD)

plt.legend()

# %%
