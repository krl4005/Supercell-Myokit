import myokit
import matplotlib.pyplot as plt
import time
import numpy as np

def plot_tor_ord():
    t = time.time()
    mod, proto, x = myokit.load('./tor_ord_endo.mmt')
    sim = myokit.Simulation(mod, proto)
    dat_normal = sim.run(10000)
    print(f'It took {time.time() - t} s')

    mod['multipliers']['i_cal_pca_multiplier'].set_rhs(8)
    sim = myokit.Simulation(mod, proto)
    dat_ical_scale = sim.run(10000)

    fig, ax = plt.subplots(1, 1, True, figsize=(12, 8))
    t = np.concatenate((dat_normal['engine.time'], np.array(dat_ical_scale['engine.time']) + 10000))
    v = np.concatenate((dat_normal['membrane.v'], dat_ical_scale['membrane.v']))

    ax.plot(t, v)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Voltage (mV)', fontsize=14)
    plt.show()


plot_tor_ord()
