#%% 
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np

conductances = [1, 1.5, 2, 2.5]

fast_v = []
fast_t = []
slow_v = []
slow_t = []

for conduct in conductances: 
    t = time.time()
    mod,proto, x = myokit.load('./tor_ord_endo.mmt')
    mod['multipliers']['i_cal_pca_multiplier'].set_rhs(conduct)
    proto.schedule(5.3, 1.1, 1, 5000, 0)
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000*5000)
    dat = sim.run(1000)

    slow_v.append(dat['membrane.v'])
    slow_t.append(dat['engine.time'])

    mod2, proto2, x = myokit.load('./tor_ord_endo.mmt')
    proto2.schedule(5.3, 1.1, 1, 500, 0)
    mod2['multipliers']['i_cal_pca_multiplier'].set_rhs(conduct)
    sim2 = myokit.Simulation(mod2, proto2)
    sim2.pre(1000*500)
    dat2 = sim2.run(500)

    fast_v.append(dat2['membrane.v'].tolist())
    fast_t.append(dat2['engine.time'].tolist())

for i in list(range(0,len(fast_t))):
    fast_label = 'fast {}'
    plt.plot(fast_t[i], fast_v[i], label = fast_label.format(i))
    slow_label = 'slow {}'
    plt.plot(slow_t[i], slow_v[i], label = slow_label.format(i))
plt.legend()
plt.xlim(-10,500)
plt.show()



# %%
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np

#t = time.time()
cl = 500
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
#mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1)
proto.schedule(5.3, 0.1, 1, cl, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(1000*cl)
dat = sim.run(1000)

#plt.plot(dat['engine.time'], dat['membrane.v'])
plt.plot(dat['engine.time'], dat['intracellular_ions.cai'])
# %%
