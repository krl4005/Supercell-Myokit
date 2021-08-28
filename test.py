#%%
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 

t = time.time()

## NORMAL
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
proto.schedule(5.3, 0.1, 1, 1000, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(1000 * 1000) #pre-pace for 1000 beats 
dat = sim.run(5000)
plt.plot(dat['engine.time'],dat['membrane.v'])
IC = mod.state()

## EAD CHALLENGE: ICaL = 8x using set_state 
mod.set_state(IC)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(8)
#proto.schedule(5.3, 0.1, 1, 1000, 0)
sim1 = myokit.Simulation(mod, proto) 
dat1 = sim1.run(5000)

plt.plot(dat1['engine.time'],dat1['membrane.v'])

## EAD CHALLENGE: ICaL = 8x without using set state
mod2, proto2, x2 = myokit.load('./tor_ord_endo.mmt')
mod2['multipliers']['i_cal_pca_multiplier'].set_rhs(8)
proto2.schedule(5.3, 0.1, 1, 1000, 0)
sim2 = myokit.Simulation(mod2, proto2)
sim2.pre(1000 * 1000) #pre-pace for 1000 beats 
dat2 = sim2.run(5000)
plt.plot(dat2['engine.time'],dat2['membrane.v'])

## EAD CHALLENGE: ICaL = 8x using set state and prepacing
mod.set_state(IC)
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(8)
#proto.schedule(5.3, 0.1, 1, 1000, 0)
sim3 = myokit.Simulation(mod, proto) 
sim3.pre(1000 * 1000) #pre-pace for 1000 beats
dat3 = sim3.run(5000)

plt.plot(dat3['engine.time'],dat3['membrane.v'])
