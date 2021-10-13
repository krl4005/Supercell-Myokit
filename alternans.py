#%%
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas
from scipy.signal import find_peaks # pip install scipy 

t = time.time()
mod, proto, x = myokit.load('./tor_ord_endo.mmt')

## ALTERNANS CHALLENGE
mod['extracellular']['cao'].set_rhs(2)
proto.schedule(5.3, 0.1, 1, 270, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(270 * 1000) #pre-pace for 1000 beats 
dat = sim.run(5000)

v = dat['membrane.v']
t = dat['engine.time']
c = dat['intracellular_ions.cai']

fig, axs = plt.subplots(2)
axs[0].plot(dat['engine.time'], dat['membrane.v'], label = "AP")
axs[1].plot(dat['engine.time'], dat['intracellular_ions.cai'], label = "Cal Trans")
fig.suptitle('Alternans Challenge', fontsize=16)

# %% Alternans Detection 
i_stim=np.array(dat['stimulus.i_stim'].tolist())
AP_S_where = np.where(i_stim!=0)[0]
AP_S_diff = np.where(np.diff(AP_S_where)!=1)[0]+1
peaks = AP_S_where[AP_S_diff]


AP = np.split(v, peaks)
t1 = np.split(t, peaks)

APD = []

for n in list(range(0, len(AP))):
    mdp = min(AP[n])
    max_p = max(AP[n])
    max_p_idx = np.argmax(AP[n])
    apa = max_p - mdp

    repol_pot = max_p - (apa * 90/100)
    idx_apd = np.argmin(np.abs(AP[n][max_p_idx:] - repol_pot))
    apd_val = t1[n][idx_apd+max_p_idx]-t1[n][0]
    APD.insert(n, apd_val) 

print(APD)


# %% Alternans Detection - Calcium 
i_stim = dat['stimulus.i_stim']
peaks = find_peaks(-np.array(i_stim), distance=100)[0]
#end_ap_idx = peaks.tolist()
cal_peak = np.split(c, peaks)
t1 = np.split(t, peaks)
#end_ap_idx.insert(len(AP), len(v))
max_cal = []

for n in list(range(0, len(cal_peak))):
    
    cal_val = max(cal_peak[n])
    max_cal.insert(n, cal_val) 

scaled_cal = [val * 1000 for val in max_cal]
print(scaled_cal)

# %%
