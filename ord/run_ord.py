#%% 
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np

#Single Run
mod, proto, x = myokit.load('./ord.mmt')
proto.schedule(1, 500, 1, 1000, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(1000*100)
dat = sim.run(1400)

t = dat['environment.time']
v = dat['membrane.v']

fig, ax = plt.subplots(1, 1, True, figsize=(12, 8))
ax.plot(t, v, color = "r")
ax.set_xlabel('Time (ms)', fontsize=14)
ax.set_ylabel('Voltage (mV)', fontsize=14)
ax.axis("off")


# %%
