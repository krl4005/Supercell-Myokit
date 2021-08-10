#%%
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 

t = time.time()
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
sim0 = myokit.Simulation(mod, proto)

dat0 = sim0.run(10000)

# EAD CHALLENGE: set ICaL = 8x
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(8)
sim1 = myokit.Simulation(mod, proto)
dat1 = sim1.run(30000)


print(f'It took {time.time() - t} s')

# Normal APs
plt.plot(dat0['engine.time'], dat0['membrane.v'], label = "Normal")

#EAD Challenge 
plt.plot(np.array(dat1['engine.time'])+10000, dat1['membrane.v'], label = "ICaL = 8")
plt.legend()
plt.show()


# %%
