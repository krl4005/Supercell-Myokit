# %%
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
path = 'cluster/fit+ead/iter3/g50_p200_e1'
#individuals = pickle.load(open("individuals", "rb"))
pop = pd.read_csv(path + '/pop.csv')
error = pd.read_csv(path + '/error.csv')

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
plt.savefig(path + '/error.png')
plt.show()

#%% 
plt.scatter(list(range(0,len(error_col))), bests, label = 'best')
plt.legend()
plt.savefig(path + '/best.png')
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
plt.savefig(path + '/last_gen.png')
plt.show()


# %%
min_index = np.where(error['gen1']==min(error['gen1']))

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

print('parameters for optimized AP:')
print('ical:', i_cal_val)
print('iks:', i_ks_val)
print('ikr:', i_kr_val)
print('inal:', i_nal_val)
print('jup:', jup_val)

# Tor-ord Baseline
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
proto.schedule(5.3, 0.1, 1, 1000, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(100*1000)
dat = sim.run(1000)

# Optimized Model
mod1, proto1, x1 = myokit.load('./tor_ord_endo.mmt')
mod1['multipliers']['i_cal_pca_multiplier'].set_rhs(i_cal_val)
mod1['multipliers']['i_kr_multiplier'].set_rhs(i_kr_val)
mod1['multipliers']['i_ks_multiplier'].set_rhs(i_ks_val)
mod1['multipliers']['i_nal_multiplier'].set_rhs(i_nal_val)
mod1['multipliers']['jup_multiplier'].set_rhs(jup_val)
proto1.schedule(5.3, 0.1, 1, 1000, 0)
sim1 = myokit.Simulation(mod1, proto1)
sim1.pre(100*1000)
dat1 = sim1.run(1000)

plt.plot(dat['engine.time'], dat['membrane.v'], label = 'baseline Tor-ord')
plt.plot(dat1['engine.time'], dat1['membrane.v'], label = 'Immunized cell')
plt.legend()
plt.savefig(path + '/AP.png')
plt.show()

print('The error val associted with this predicted AP is:', min(error[error_col[-1]]))

# %% Calculate ap_features
cl = 1000
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(i_cal_val)
mod['multipliers']['i_kr_multiplier'].set_rhs(i_kr_val)
mod['multipliers']['i_ks_multiplier'].set_rhs(i_ks_val)
mod['multipliers']['i_nal_multiplier'].set_rhs(i_nal_val)
mod['multipliers']['jup_multiplier'].set_rhs(jup_val)
proto.schedule(5.3, 0.1, 1, cl, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(100*cl)
dat = sim.run(5000)
IC = sim.state()

# Get t, v, and cai for second to last AP#######################
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

ap_features = {}

# Voltage/APD features#######################
mdp = min(v)
max_p = max(v)
max_p_idx = np.argmax(v)
apa = max_p - mdp
dvdt_max = np.max(np.diff(v[0:30])/np.diff(t[0:30]))

ap_features['dvdt_max'] = dvdt_max

for apd_pct in [10, 50, 90]:
    repol_pot = max_p - apa * apd_pct/100
    idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
    apd_val = t[idx_apd+max_p_idx]
    
    if apd_pct == 10:
        ap_features[f'apd{apd_pct}'] = apd_val*5
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

print(ap_features)

# %% Challenge - ICaL
# Tor-ord Baseline
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(8)
proto.schedule(5.3, 0.1, 1, 1000, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(100*1000)
dat = sim.run(1000)

# Optimized Model
mod1, proto1, x1 = myokit.load('./tor_ord_endo.mmt')
mod1['multipliers']['i_cal_pca_multiplier'].set_rhs(i_cal_val*8)
mod1['multipliers']['i_kr_multiplier'].set_rhs(i_kr_val)
mod1['multipliers']['i_ks_multiplier'].set_rhs(i_ks_val)
mod1['multipliers']['i_nal_multiplier'].set_rhs(i_nal_val)
mod1['multipliers']['jup_multiplier'].set_rhs(jup_val)
proto1.schedule(5.3, 0.1, 1, 1000, 0)
sim1 = myokit.Simulation(mod1, proto1)
sim1.pre(100*1000)
dat1 = sim1.run(1000)

plt.plot(dat['engine.time'], dat['membrane.v'], label = 'Tor-ord')
plt.plot(dat1['engine.time'], dat1['membrane.v'], label = 'Immunized cell')
plt.legend()
plt.savefig(path + '/chal_ical.png')
plt.show()

# %% Challenge - IKr
# Tor-ord Baseline
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
mod['multipliers']['i_kr_multiplier'].set_rhs(0.01)
proto.schedule(5.3, 0.1, 1, 2500, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(100*2500)
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
sim1.pre(100*2500)
dat1 = sim1.run(1000)

plt.plot(dat['engine.time'], dat['membrane.v'], label = 'Tor-ord')
plt.plot(dat1['engine.time'], dat1['membrane.v'], label = 'Immunized cell')
plt.legend()
plt.savefig(path + '/chal_ikr.png')
plt.show()

# %%
