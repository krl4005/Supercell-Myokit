#%%
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 

t0 = time.time()

## RRC CHALLENGE
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
proto.schedule(5.3, 0.1, 1, 1000, 0)
proto.schedule(0.025, 4, 995, 1000, 1)
proto.schedule(0.05, 5004, 995, 1000, 1)
proto.schedule(0.075, 10004, 995, 1000, 1)
proto.schedule(0.1, 15004, 995, 1000, 1)
proto.schedule(0.125, 20004, 995, 1000, 1)
sim = myokit.Simulation(mod, proto)
dat = sim.run(25000)

print(f'It took {time.time() - t0} s')

fig, axs = plt.subplots(2, constrained_layout=True, figsize=(15,7))
axs[0].plot(dat['engine.time'], dat['membrane.v'])
axs[1].plot(dat['engine.time'], dat['intracellular_ions.cai'])
fig.suptitle('RRC Challenge', fontsize=16)

plt.show()

# %% Pull out APs with RRC stimulus 
v = dat['membrane.v']
t = dat['engine.time']

i_stim=np.array(dat['stimulus.i_stim'].tolist())
AP_S_where = np.where(i_stim== -53.0)[0]
AP_S_diff = np.where(np.diff(AP_S_where)!=1)[0]+1
peaks = AP_S_where[AP_S_diff]

AP = np.split(v, peaks)
t1 = np.split(t, peaks)
s = np.split(i_stim, peaks)
print(len(AP))

# %% EAD Detection 
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

print(vals)

# %% RRC DETECTION
stims = []
for v in list(range(0, len(vals))): 
    RRC = s[v][160]
    stims.insert(v, RRC)

print(stims) 

for v in list(range(0, len(vals))): 
    if vals[v] > 1:
        RRC = s[v][160]
        break

print(RRC) 

# %%
