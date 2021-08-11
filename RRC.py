#%%
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 

t = time.time()
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
proto.schedule(5.3, 0.1, 1, 1000, 0)

## RRC CHALLENGE
proto.schedule(0.025, 4, 995, 1000, 1)
proto.schedule(0.05, 5004, 995, 1000, 1)
proto.schedule(0.075, 10004, 995, 1000, 1)
proto.schedule(0.1, 15004, 995, 1000, 1)
proto.schedule(0.125, 20004, 995, 1000, 1)
sim = myokit.Simulation(mod, proto)
dat = sim.run(25000)

fig, axs = plt.subplots(2)
axs[0].plot(dat['engine.time'], dat['membrane.v'], label = "AP")
axs[1].plot(dat['engine.time'], dat['intracellular_ions.cai'], label = "Cal Trans")
fig.suptitle('RRC Challenge', fontsize=16)
# %%
