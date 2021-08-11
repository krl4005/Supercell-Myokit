#%%
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
# %%
