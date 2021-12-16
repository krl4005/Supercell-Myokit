#%%
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 

t = time.time()

## NORMAL
mod0, proto0, x0 = myokit.load('./tor_ord_endo.mmt')
proto0.schedule(5.3, 0.1, 1, 1000, 0)
sim0 = myokit.Simulation(mod0, proto0)
dat0 = sim0.run(5000)

## ALTERNANS CHALLENGE
mod1, proto1, x1 = myokit.load('./tor_ord_endo.mmt')
mod1['extracellular']['cao'].set_rhs(2)
proto1.schedule(5.3, 0.1, 1, 270, 0)
sim1 = myokit.Simulation(mod1, proto1)
sim1.pre(500 * 1000) #pre-pace for 500 beats 
dat1 = sim1.run(1000)

## EAD CHALLENGE: ICaL = 8x
mod2, proto2, x2 = myokit.load('./tor_ord_endo.mmt')
mod2['multipliers']['i_cal_pca_multiplier'].set_rhs(8)
proto2.schedule(5.3, 0.1, 1, 1000, 0)
sim2 = myokit.Simulation(mod2, proto2)
sim2.pre(100 * 1000) #pre-pace for 100 beats 
dat2 = sim2.run(1000)

## RRC CHALLENGE
mod3, proto3, x3 = myokit.load('./tor_ord_endo.mmt')
proto3.schedule(5.3, 0.1, 1, 1000, 0)
proto3.schedule(0.025, 4, 995, 1000, 1)
proto3.schedule(0.05, 5004, 995, 1000, 1)
proto3.schedule(0.075, 10004, 995, 1000, 1)
proto3.schedule(0.1, 15004, 995, 1000, 1)
proto3.schedule(0.125, 20004, 995, 1000, 1)
sim3 = myokit.Simulation(mod3, proto3)
dat3 = sim3.run(25000)

print(f'It took {time.time() - t} s')

# Plots
fig, axs = plt.subplots(6, constrained_layout=True, figsize=(15,15))
fig.suptitle('Supercell Protocol', fontsize=25)

axs[0].set_title('Normal Beats')
axs[0].plot(dat0['engine.time'], dat0['membrane.v'], label = "Normal")

axs[1].set_title('Alternan Challenge')
axs[1].plot(dat1['engine.time'], dat1['membrane.v'], label = "Alternans")

axs[2].set_title('EAD Challenge: ICal=8')
axs[2].plot(dat2['engine.time'], dat2['membrane.v'], label = "EAD")

axs[3].set_title('RRC Challenge: -0.25, -0.5, -0.75, -1.0, -1.25')
axs[3].plot(dat3['engine.time'], dat3['membrane.v'], label = "RRC")

axs[4].set_title('Alternan Challenge: Calcium Transient')
axs[4].plot(dat1['engine.time'], dat1['intracellular_ions.cai'], label = "Cal Trans")

t = np.concatenate((dat0['engine.time'], np.array(dat1['engine.time']) + 5000, np.array(dat0['engine.time']) + 8000, np.array(dat2['engine.time'])+13000, np.array(dat3['engine.time'])+14000))
v = np.concatenate((dat0['membrane.v'], dat1['membrane.v'], dat0['membrane.v'], dat2['membrane.v'], dat3['membrane.v']))
axs[5].set_title('Full Protocol')
axs[5].plot(t, v, label = "Full Protocol")
plt.show()


# %%
