#%% 
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np

conductances = [1]

fast_v = []
fast_t = []
slow_v = []
slow_t = []

for conduct in conductances: 
    t = time.time()
    mod,proto,x = myokit.load('./TP06.mmt')
    mod['multipliers']['i_cal_pca_multiplier'].set_rhs(conduct)
    proto.schedule(1, 1, 1, 5000, 0)
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000*5000)
    dat = sim.run(1000)

    slow_v.append(dat['membrane.V'])
    slow_t.append(dat['environment.time'])

    mod2, proto2,x = myokit.load('./TP06.mmt')
    proto2.schedule(1, 1, 1, 500, 0)
    mod2['multipliers']['i_cal_pca_multiplier'].set_rhs(conduct)
    sim2 = myokit.Simulation(mod2, proto2)
    sim2.pre(1000*500)
    dat2 = sim2.run(500)

    fast_v.append(dat2['membrane.V'].tolist())
    fast_t.append(dat2['environment.time'].tolist())

# TP06
for i in list(range(0,len(fast_t))):
    fast_label = 'fast {}'
    plt.plot(fast_t[i], fast_v[i], label = fast_label.format(i))
    slow_label = 'slow {}'
    plt.plot(slow_t[i], slow_v[i], label = slow_label.format(i))
plt.xlim(-10,500)
plt.legend()
plt.show()


# %%
#mod,proto,x = myokit.load('./TP06.mmt')
mod,proto,x = myokit.load('./TP06.mmt')
#proto = myokit.Protocol()
#mod['multipliers']['i_cal_pca_multiplier'].set_rhs(conduct)
proto.schedule(1, 1, 1, 5000, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(1000*5000)
dat = sim.run(1000)

plt.plot(dat['environment.time'], dat['membrane.V'])
plt.xlim(-10,500)

# %%
