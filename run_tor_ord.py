#%% 
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np

def plot_kernik_mc():
    t = time.time()
    mod, proto, x = myokit.load('./tor_ord_endo.mmt')
    sim = myokit.Simulation(mod, proto)
    dat = sim.run(10000)
    print(f'It took {time.time() - t} s')

    fig, axs = plt.subplots(2, 1, True)
    axs[0].plot(dat['engine.time'], dat['membrane.v'])
    axs[1].plot(dat['engine.time'], dat['membrane.i_ion'])

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    plt.show()

plot_kernik_mc()

# %%
