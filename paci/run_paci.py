

#%%
#single run
import myokit
import matplotlib.pyplot as plt

mod,proto, x = myokit.load('./paci2013_ventricular.mmt')
#mod['i_CaL']['g_CaL'].set_rhs(mod['i_CaL']['g_CaL'].value()*3)
proto.schedule(0, 1, 1, 1, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(1*100)
dat = sim.run(4)

t = dat['environment.time']
v = dat['Membrane.Vm']

plt.plot(t, v, color = "r")
#ax.set_xlabel('Time (s)', fontsize=14)
#ax.set_ylabel('Voltage (V)', fontsize=14)
plt.xlim([1,3])
plt.axis("off")

# %%
