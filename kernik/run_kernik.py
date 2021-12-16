#%% 
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np


t = time.time()

#baseline
mod, proto, x = myokit.load('./kernik.mmt')
proto.schedule(0, 100, 5, 1000, 0)
sim = myokit.Simulation(mod, proto)
dat_normal = sim.run(5000)
#print(f'It took {time.time() - t} s')
t = dat_normal['engine.time']
v = dat_normal['membrane.V']

#mature
modm, protom, xm = myokit.load('./kernik.mmt')
protom.schedule(1, 500, 5, 1000, 0)
modm['ik1']['g_K1'].set_rhs(modm['ik1']['g_K1'].value()*(11.24/5.67))
modm['ina']['g_Na'].set_rhs(modm['ina']['g_Na'].value()*(187/129))
#modm['engine']['pace'].set_rhs(-1)
simm = myokit.Simulation(modm, protom)
simm.pre(100 * 1000) #pre-pace for 100 beats 
dat_mature = simm.run(5000)
#print(f'It took {time.time() - t} s')
tm = dat_mature['engine.time']
vm = dat_mature['membrane.V']

fig, ax = plt.subplots(1, 1, True, figsize=(12, 8))
ax.plot(t, v)
ax.plot(tm,vm)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Time (ms)', fontsize=14)
ax.set_ylabel('Voltage (mV)', fontsize=14)
plt.show()

# %%
