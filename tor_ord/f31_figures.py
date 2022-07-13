#%% Run Simulation and Plot AP
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 
from scipy.signal import find_peaks # pip install scipy

#%% EAD/STIM FIGURE CODE
t = time.time()
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
proto.schedule(5.3, 0, 1, 1000, 0)

## EAD CHALLENGE: ICaL = 8x
#mod['multipliers']['i_cal_pca_multiplier'].set_rhs(8)

## EAD CHALLENGE: Istim = -.1
proto.schedule(0.1, 2004, 1000-100, 1000, 1)

sim = myokit.Simulation(mod, proto)
sim.pre(100 * 1000) #pre-pace for 100 beats 
dat = sim.run(5000)

fig, axs = plt.subplots(2)
axs[0].plot(dat['engine.time'], dat['membrane.v'], label = "AP")
axs[0].set_xlim([-100,3000])
axs[0].set_ylabel("Voltage (mV)")
axs[1].plot(dat['engine.time'], dat['stimulus.i_stim'], label = "Current")
axs[1].set_xlim([-100,3000])
axs[1].set_xlabel("Time (ms)")
axs[1].set_ylabel("Stimulus (A/F)")
#fig.suptitle('EAD Challenge', fontsize=16)
plt.show()

#%% EAD/STIM FIGURE CODE
plt.figure(figsize=[5,2])
plt.plot(dat['engine.time'], dat['stimulus.i_stim'])
plt.ylim([-5, 1])
plt.xlim([-100, 3000])
plt.xlabel("Time (t)")
plt.ylabel("Stimulus (A/F)")