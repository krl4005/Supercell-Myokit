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

# READ IN DATA 
path = 'c:\\Users\\Kristin\\Desktop\\Christini Lab\\Research Data\\supercell-myokit\\cluster\\fit+ead\\Istim\\iter2\\g50_p200_e2\\trial3'
#individuals = pickle.load(open("individuals", "rb"))
pop = pd.read_csv(path + '\\pop.csv')
error = pd.read_csv(path + '\\error.csv')

#%%
pop_col = pop.columns.tolist()
last_gen = []

for i in list(range(0, len(pop[pop_col[-1]]))):
    last_gen.append(literal_eval(pop[pop_col[-1]][i]))

i_cal = []
i_ks = []
i_kr = []
i_nal = []
jup = []

for i in list(range(0,len(last_gen))):
    i_cal.append(last_gen[i][0])
    i_ks.append(last_gen[i][1]) 
    i_kr.append(last_gen[i][2])
    i_nal.append(last_gen[i][3])  
    jup.append(last_gen[i][4]) 

#%%  
error_col = error.columns.tolist()
avgs = []
bests = []

for i in list(range(0, len(error_col))):
    avg = sum(error[error_col[i]])/len(error[error_col[i]])
    avgs.append(avg)
    best = min(error[error_col[i]]) 
    bests.append(best)

plt.scatter(list(range(0,len(error_col))), avgs, label = 'average')
plt.scatter(list(range(0,len(error_col))), bests, label = 'best')
plt.legend()
plt.savefig(path + '\\error.png')
plt.show()

#%% 
plt.scatter(list(range(0,len(error_col))), bests, label = 'best')
plt.xlabel("Generation Number")
plt.ylabel("Error")
plt.savefig(path + '\\best.png')
plt.show()


# %%
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter([1]*len(i_cal),i_cal, c=error[error_col[-1]], cmap=cm)
sc = plt.scatter([2]*len(i_ks),i_ks, c=error[error_col[-1]], cmap=cm)
sc = plt.scatter([3]*len(i_kr),i_kr, c=error[error_col[-1]], cmap=cm)
sc = plt.scatter([4]*len(i_nal),i_nal, c=error[error_col[-1]], cmap=cm)
sc = plt.scatter([5]*len(jup),jup, c=error[error_col[-1]], cmap=cm)

plt.colorbar(sc)
positions = (1, 2, 3, 4, 5)
label = ('GCaL', 'GKs', 'GKr', 'GNaL', 'Jup')
plt.ylabel("Conductance Value")
plt.xticks(positions, label)
plt.savefig(path + '\\last_gen.png')
plt.show()

# %%
cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter([1]*len(i_cal),i_cal, c=error[error_col[-1]], cmap=cm)
sc = plt.scatter([2]*len(i_ks),i_ks, c=error[error_col[-1]], cmap=cm)
sc = plt.scatter([3]*len(i_kr),i_kr, c=error[error_col[-1]], cmap=cm)
sc = plt.scatter([4]*len(i_nal),i_nal, c=error[error_col[-1]], cmap=cm)
sc = plt.scatter([5]*len(jup),jup, c=error[error_col[-1]], cmap=cm)

plt.colorbar(sc)
positions = (1, 2, 3, 4, 5)
label = ('GCaL', 'GKs', 'GKr', 'GNaL', 'Jup')
plt.xticks(positions, label)
plt.ylim([-1, 10])
plt.ylabel("Conductance Value")
plt.savefig(path + '\\last_gen_scale.png')
plt.show()

#%% 
def get_last_ap(dat):
    
    i_stim = dat['stimulus.i_stim']

    # This is here so that stim for EAD doesnt interfere with getting the whole AP
    for i in list(range(0,len(i_stim))):
        if abs(i_stim[i])<50:
            i_stim[i]=0

    peaks = find_peaks(-np.array(i_stim), distance=100)[0]
    start_ap = peaks[-3] #TODO change start_ap to be after stim, not during
    end_ap = peaks[-2]

    print(start_ap, dat['engine.time'][start_ap])
    print(end_ap, dat['engine.time'][end_ap])

    t = np.array(dat['engine.time'][start_ap:end_ap])
    t = t - t[0]
    max_idx = np.argmin(np.abs(t-900))
    t = t[0:max_idx]
    end_ap = start_ap + max_idx

    v = np.array(dat['membrane.v'][start_ap:end_ap])
    cai = np.array(dat['intracellular_ions.cai'][start_ap:end_ap])
    i_ion = np.array(dat['membrane.i_ion'][start_ap:end_ap])

    return (t, v, cai, i_ion)

def get_ead_error(mod, proto, sim, ind, s): 
    #mod['multipliers']['i_cal_pca_multiplier'].set_rhs(ind[0]['i_cal_pca_multiplier']*8)
    proto.schedule(s, 2004, 1000-100, 1000, 1)
    sim = myokit.Simulation(mod, proto)
    dat = sim.run(5000)
    #plt.plot(dat['engine.time'], dat['membrane.v'])

    ########### EAD DETECTION ############# 
    t,v,cai,i_ion = get_last_ap(dat)

    start = 100
    start_idx = np.argmin(np.abs(np.array(t)-start)) #find index closest to t=100
    end_idx = len(t)-3 #subtract 3 because we are stepping by 3 in the loop

    t_idx = list(range(start_idx, end_idx)) 

    # Find rises in the action potential after t=100
    rises = []
    for t in t_idx:
        v1 = v[t]
        v2 = v[t+3]

        if v2>v1:
            rises.insert(t,v2)
        else:
            rises.insert(t,0)

    for i in list(range(0,len(rises))):
        if 80-abs(rises[i]) < 1:
            cutoff = i
            for z in list(range(cutoff, len(rises)-1)):
                rises[z]=0

    if np.count_nonzero(rises) != 0: 
        # Pull out blocks of rises 
        rises=np.array(rises)
        EAD_idx = np.where(rises!=0)[0]
        diff_idx = np.where(np.diff(EAD_idx)!=1)[0]+1 #index to split rises at
        EADs = np.split(rises[EAD_idx], diff_idx)

        amps = []
        E_idx = list(range(0, len(EADs))) 
        for x in E_idx:
            low = min(EADs[x])
            high = max(EADs[x])

            a = high-low
            amps.insert(x, a) 

        EAD = max(amps)
        EAD_val = EADs[np.where(amps==max(amps))[0][0]]

    else:
        EAD = 0
    return EAD, dat, rises

def get_feature_errors(sim):
    t,v,cai,i_ion = get_normal_sim_dat(sim)

    ap_features = {}

    # Returns really large error value if cell AP is not valid 
    if ((min(v) > -60) or (max(v) < 0)):
        return 500000 

    # Voltage/APD features#######################
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    dvdt_max = np.max(np.diff(v[0:30])/np.diff(t[0:30]))

    ap_features['dvdt_max'] = dvdt_max

    for apd_pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        repol_pot = max_p - apa * apd_pct/100
        idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
        apd_val = t[idx_apd+max_p_idx]
        
        if apd_pct == 10 or apd_pct == 40:
            ap_features[f'apd{apd_pct}'] = apd_val*5 #scale APD10 and APD40
        else:
            ap_features[f'apd{apd_pct}'] = apd_val

    # Calcium/CaT features######################## 
    max_cai = np.max(cai)
    max_cai_idx = np.argmax(cai)
    cat_amp = np.max(cai) - np.min(cai)
    ap_features['cat_amp'] = cat_amp * 1e5 #added in multiplier since number is so small

    for cat_pct in [10, 50, 90]:
        cat_recov = max_cai - cat_amp * cat_pct / 100
        idx_catd = np.argmin(np.abs(cai[max_cai_idx:] - cat_recov))
        catd_val = t[idx_catd+max_cai_idx]

        ap_features[f'cat{cat_pct}'] = catd_val 

    return ap_features

def get_normal_sim_dat(sim):
    # Get t, v, and cai for second to last AP#######################
    dat = sim.run(5000)
    i_stim = dat['stimulus.i_stim']
    peaks = find_peaks(-np.array(i_stim), distance=100)[0]
    start_ap = peaks[-3] #TODO change start_ap to be after stim, not during
    end_ap = peaks[-2]

    t = np.array(dat['engine.time'][start_ap:end_ap])
    t = t - t[0]
    max_idx = np.argmin(np.abs(t-900))
    t = t[0:max_idx]
    end_ap = start_ap + max_idx

    v = np.array(dat['membrane.v'][start_ap:end_ap])
    cai = np.array(dat['intracellular_ions.cai'][start_ap:end_ap])
    i_ion = np.array(dat['membrane.i_ion'][start_ap:end_ap])

    return (t, v, cai, i_ion)

def get_ind_data(ind):
    mod, proto, x = myokit.load('./tor_ord_endo.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    proto.schedule(5.3, 0.1, 1, 1000, 0) #ADDED IN
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats 
    IC = sim.state()

    return mod, proto, sim, IC

# %%
# USE CODE BELOW TO LOOK AT A SPECIFIC POP
#conduct = literal_eval(pop['gen1'][161])
#i_cal_val = conduct[0]
#i_ks_val = conduct[1]
#i_kr_val = conduct[2]
#i_nal_val = conduct[3]
#jup_val = conduct[4]

min_index = np.where(error[error_col[-1]]==min(error[error_col[-1]]))
i_cal_val = i_cal[min_index[0][0]]
i_ks_val = i_ks[min_index[0][0]]
i_kr_val = i_kr[min_index[0][0]]
i_nal_val = i_nal[min_index[0][0]]
jup_val = jup[min_index[0][0]]

tunable_parameters=['i_cal_pca_multiplier',
                    'i_ks_multiplier',
                    'i_kr_multiplier',
                    'i_nal_multiplier',
                    'jup_multiplier'],

base = [1, 1, 1, 1, 1]

opt = [i_cal_val, i_kr_val, i_ks_val, i_nal_val, jup_val]

keys = [val for val in tunable_parameters]
baseline = [dict(zip(keys[0], base))]
optimized = [dict(zip(keys[0], opt))]

print('parameters for optimized AP:', optimized)
print('baseline:', baseline)

# Tor-ord Baseline
mod, proto, sim, IC = get_ind_data(baseline)
dat = sim.run(5000)

# Optimized Model
mod1, proto1, sim1, IC1 = get_ind_data(optimized)
dat1 = sim1.run(5000)

plt.plot(dat['engine.time'], dat['membrane.v'], label = 'baseline cell')
plt.plot(dat1['engine.time'], dat1['membrane.v'], label = 'resistant cell')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.xlim([-100,1000])
plt.legend()
plt.savefig(path + '\\AP.png')
plt.show()

print('The error val associted with this predicted AP is:', min(error[error_col[-1]]))

# %% Calculate ap_features
features_opt = get_feature_errors(sim1)
print("     ")
print(features_opt)

# %% Challenge - stimulus
mod2, proto2, sim2, IC2 = get_ind_data(baseline)
EAD2, dat2, rises2 = get_ead_error(mod2, proto2, sim2, baseline, 0.1)

mod3, proto3, sim3, IC3 = get_ind_data(optimized)
EAD3, dat3, rises3 = get_ead_error(mod3, proto3, sim3, optimized, 0.1)


plt.plot(dat2['engine.time'], dat2['membrane.v'], label = 'baseline cell')
plt.plot(dat3['engine.time'], dat3['membrane.v'], label = 'resistant cell')
plt.xlim([1800, 3000])
plt.legend()
plt.savefig(path + '\\chal_stim.png')
plt.show()

# %% Challenge - ICaL
# Tor-ord Baseline
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(6)
proto.schedule(5.3, 0.1, 1, 1000, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(100*1000)
dat = sim.run(1000)

# Optimized Model
mod1, proto1, x1 = myokit.load('./tor_ord_endo.mmt')
mod1['multipliers']['i_cal_pca_multiplier'].set_rhs(i_cal_val*6)
mod1['multipliers']['i_kr_multiplier'].set_rhs(i_kr_val)
mod1['multipliers']['i_ks_multiplier'].set_rhs(i_ks_val)
mod1['multipliers']['i_nal_multiplier'].set_rhs(i_nal_val)
mod1['multipliers']['jup_multiplier'].set_rhs(jup_val)
proto1.schedule(5.3, 0.1, 1, 1000, 0)
sim1 = myokit.Simulation(mod1, proto1)
sim1.pre(100*1000)
dat1 = sim1.run(1000)

plt.plot(dat['engine.time'], dat['membrane.v'], label = 'baseline cell')
plt.plot(dat1['engine.time'], dat1['membrane.v'], label = 'resistant cell')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.savefig(path + '\\chal_ical.png')
plt.show()

# %% Challenge - IKr
# Tor-ord Baseline
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
mod['multipliers']['i_kr_multiplier'].set_rhs(0.01)
proto.schedule(5.3, 0.1, 1, 2500, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(100*1000)
dat = sim.run(1000)

# Optimized Model
mod1, proto1, x1 = myokit.load('./tor_ord_endo.mmt')
mod1['multipliers']['i_cal_pca_multiplier'].set_rhs(i_cal_val)
mod1['multipliers']['i_kr_multiplier'].set_rhs(i_kr_val*0.01)
mod1['multipliers']['i_ks_multiplier'].set_rhs(i_ks_val)
mod1['multipliers']['i_nal_multiplier'].set_rhs(i_nal_val)
mod1['multipliers']['jup_multiplier'].set_rhs(jup_val)
proto1.schedule(5.3, 0.1, 1, 2500, 0)
sim1 = myokit.Simulation(mod1, proto1)
sim1.pre(100*1000)
dat1 = sim1.run(1000)

plt.plot(dat['engine.time'], dat['membrane.v'], label = 'baseline cell')
plt.plot(dat1['engine.time'], dat1['membrane.v'], label = 'resistant cell')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.savefig(path + '\\chal_ikr.png')
plt.show()

# %% RRC FUNCTIONS

def get_rrc_error(mod, proto, sim):

    ## RRC CHALLENGE
    proto.schedule(5.3, 0.2, 1, 1000, 0)
    proto.schedule(0.025, 4, 995, 1000, 1)
    proto.schedule(0.05, 5004, 995, 1000, 1)
    proto.schedule(0.075, 10004, 995, 1000, 1)
    proto.schedule(0.1, 15004, 995, 1000, 1)
    proto.schedule(0.125, 20004, 995, 1000, 1)

    sim = myokit.Simulation(mod, proto)
    sim.pre(100 * 1000) #pre-pace for 100 beats
    dat = sim.run(25000)

    # Pull out APs with RRC stimulus 
    v = dat['membrane.v']
    t = dat['engine.time']

    i_stim=np.array(dat['stimulus.i_stim'].tolist())
    AP_S_where = np.where(i_stim== -53.0)[0]
    AP_S_diff = np.where(np.diff(AP_S_where)!=1)[0]+1
    peaks = AP_S_where[AP_S_diff]

    AP = np.split(v, peaks)
    t1 = np.split(t, peaks)
    s = np.split(i_stim, peaks)

    ########### EAD DETECTION ############# 
    vals = []

    for n in list(range(0, len(AP))): 

        AP_t = t1[n]
        AP_v = AP[n] 

        start = 100 + (1000*n)
        start_idx = np.argmin(np.abs(np.array(AP_t)-start)) #find index closest to t=100
        end_idx = len(AP_t)-3 #subtract 3 because we are stepping by 3 in the loop

        # Find rises in the action potential after t=100 
        rises = []
        for x in list(range(start_idx, end_idx)):
            v1 = AP_v[x]
            v2 = AP_v[x+3]

            if v2>v1:
                rises.insert(x,v2)
            else:
                rises.insert(x,0)

        if np.count_nonzero(rises) != 0: 
            # Pull out blocks of rises 
            rises=np.array(rises)
            EAD_idx = np.where(rises!=0)[0]
            diff_idx = np.where(np.diff(EAD_idx)!=1)[0]+1 #index to split rises at
            EADs = np.split(rises[EAD_idx], diff_idx)

            amps = []
            for y in list(range(0, len(EADs))) :
                low = min(EADs[y])
                high = max(EADs[y])

                a = high-low
                amps.insert(y, a) 

            EAD = max(amps)
            EAD_val = EADs[np.where(amps==max(amps))[0][0]]

        else:
            EAD = 0

        vals.insert(n, EAD)

    #################### RRC DETECTION ###########################
    global RRC
    global E_RRC

    RRC_vals = [-0.0, -53.0, -0.25, -0.5, -0.75, -1.0, -1.25]
    error_vals = [5000, 5000, 4000, 3000, 2000, 1000, 0]
    
    for v in list(range(0, len(vals))): 
        if vals[v] > 1:
            RRC = s[v][np.where(np.diff(s[v])!=0)[0][2]]
            break
        else:
            RRC = -1.25 #if there is no EAD than the stim was not strong enough so error should be zero

    E_RRC = error_vals[RRC_vals.index(RRC)]

    return RRC, E_RRC, vals, dat, s, v

def get_ind_data(ind):
    mod, proto, x = myokit.load('./tor_ord_endo.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    proto.schedule(5.3, 0.1, 1, 1000, 0) #ADDED IN
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats 
    IC = sim.state()

    return mod, proto, sim, IC

#%% RRC Calculation - immunized
tunable_parameters=['i_cal_pca_multiplier',
                    'i_ks_multiplier',
                    'i_kr_multiplier',
                    'i_nal_multiplier',
                    'jup_multiplier'],

initial_params = [i_cal_val, 
                  i_ks_val, 
                  i_kr_val, 
                  i_nal_val, 
                  jup_val]

keys = [val for val in tunable_parameters]
final = [dict(zip(keys[0], initial_params))]

print(final[0])

mod, proto, sim, IC = get_ind_data(final)
RRC, E_RRC, vals, dat, s, v = get_rrc_error(mod, proto, sim)
print(RRC, E_RRC)

plt.figure(figsize=[10,3])
plt.plot(dat['engine.time'], dat['membrane.v'])
plt.savefig(path + '\\rrc_resistant.png')

#%% RRC Calculation - baseline
initial_params = [1, 1, 1, 1, 1]

keys = [val for val in tunable_parameters]
final = [dict(zip(keys[0], initial_params))]

print(final[0])

mod, proto, sim, IC = get_ind_data(final)
RRC, E_RRC, vals, dat, s, v = get_rrc_error(mod, proto, sim)
print(RRC, E_RRC)

plt.figure(figsize=[10,3])
#plt.plot(dat['engine.time'], dat['membrane.v'])
plt.plot(dat['engine.time'], dat['stimulus.i_stim'])
plt.ylim([-3,0.5])
plt.ylabel("Simulus (A/F)")
plt.xlabel("Time (t)")
#plt.savefig(path + '\\rrc_base.png')

# %%
