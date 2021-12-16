#%% Run Simulation and Plot AP
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 
from scipy.signal import find_peaks # pip install scipy

t = time.time()
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
proto.schedule(5.3, 0.1, 1, 1000, 0)

## EAD CHALLENGE: ICaL = 8x
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(8)
sim = myokit.Simulation(mod, proto)
sim.pre(100 * 1000) #pre-pace for 100 beats 
dat = sim.run(5000)

fig, axs = plt.subplots(2)
axs[0].plot(dat['engine.time'], dat['membrane.v'], label = "AP")
axs[1].plot(dat['engine.time'], dat['intracellular_ions.cai'], label = "Cal Trans")
fig.suptitle('EAD Challenge', fontsize=16)
plt.show()

# %% FUNCTIONS
def get_last_ap(dat):

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

    return (t, v, cai, i_ion)

# %% EAD Detection 
t,v,cai,i_ion = get_last_ap(dat)

start = 100
start_idx = np.argmin(np.abs(np.array(t)-start)) #find index closest to t=100
end_idx = len(t)-3 #subtract 3 because we are stepping by 3 in the loop

t_idx = list(range(start_idx, end_idx)) 

# Find rises in the action potential after t=100
rises = []
#rises = []
for t in t_idx:
    v1 = v[t]
    v2 = v[t+3]

    if v2>v1:
        rises.insert(t,v2)
    else:
        rises.insert(t,0)

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

print(EAD)



# %%
