#%% Run Simulation and Plot AP
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 

t = time.time()
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
proto.schedule(5.3, 0.1, 1, 1000, 0)

## EAD CHALLENGE: ICaL = 8x
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(8)
sim = myokit.Simulation(mod, proto)
sim.pre(100 * 1000) #pre-pace for 100 beats 
dat = sim.run(1000)

fig, axs = plt.subplots(2)
axs[0].plot(dat['engine.time'], dat['membrane.v'], label = "AP")
axs[1].plot(dat['engine.time'], dat['intracellular_ions.cai'], label = "Cal Trans")
fig.suptitle('EAD Challenge', fontsize=16)

# %% EAD Detection 
v = dat['membrane.v']
t = dat['engine.time']

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
