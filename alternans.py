#%%
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 

t = time.time()
mod, proto, x = myokit.load('./tor_ord_endo.mmt')

## ALTERNANS CHALLENGE
mod['extracellular']['cao'].set_rhs(2)
proto.schedule(5.3, 0.1, 1, 270, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(500 * 1000) #pre-pace for 500 beats 
dat = sim.run(10000)

fig, axs = plt.subplots(2)
axs[0].plot(dat['engine.time'], dat['membrane.v'], label = "AP")
axs[1].plot(dat['engine.time'], dat['intracellular_ions.cai'], label = "Cal Trans")
fig.suptitle('Alternans Challenge', fontsize=16)
# %%
