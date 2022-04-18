
#%% FUNCTIONS
from re import M
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 

#%%
t = time.time()
cl = 1000
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(15)
proto.schedule(5.3, 0.1, 1, cl, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(100*cl)
dat = sim.run(1000)

#vt90 = 0.9*sim.state()[mod.get('membrane.v').indice()]
#apd90 = dat.apd(v='membrane.v', threshold = vt90)
#print(apd90)

IC = sim.state()

plt.plot(dat['engine.time'], dat['membrane.v'])
#plt.plot(dat['engine.time'], dat['intracellular_ions.cai'])

#%%
def evaluate_APD90(dat, repol): 
    i_stim=np.array(dat['stimulus.i_stim'].tolist())
    AP_S_where = np.where(i_stim!=0)[0]
    AP_S_diff = np.where(np.diff(AP_S_where)!=1)[0]+1
    peaks = AP_S_where[AP_S_diff]

    v = dat['membrane.v']
    t = dat['engine.time']

    AP = np.split(v, peaks)
    t1 = np.split(t, peaks)

    APD = []

    for n in list(range(0, len(AP))):
        mdp = min(AP[n])
        max_p = max(AP[n])
        max_p_idx = np.argmax(AP[n])
        apa = max_p - mdp

        repol_pot = max_p - (apa * repol/100)
        idx_apd = np.argmin(np.abs(AP[n][max_p_idx:] - repol_pot))
        apd_val = t1[n][idx_apd+max_p_idx]-t1[n][0]
        APD.insert(n, apd_val) 

    return(APD)

#%% 
apd10 = evaluate_APD90(dat, 10)
print(apd10)
apd20 = evaluate_APD90(dat, 20)
print(apd20)
apd30 = evaluate_APD90(dat, 30)
print(apd30)
apd40 = evaluate_APD90(dat, 40)
print(apd40)
apd50 = evaluate_APD90(dat, 50)
print(apd50)
apd60 = evaluate_APD90(dat, 60)
print(apd60)
apd70 = evaluate_APD90(dat, 70)
print(apd70)
apd80 = evaluate_APD90(dat, 80)
print(apd80)
apd90 = evaluate_APD90(dat, 90)
print(apd90)

# %%

def calulate_APD(v_dat, t_dat):
    t_depol = v_dat[0].index(max(v_dat[0]))

    v_60 = []
    wrong = []
    for i in list(range(t_depol, len(v_dat[0]))):
        d = np.abs(-60-v_dat[0][i])
        if d < 1:
            v_60.append(i)
        else:
            wrong.append(i)

    t_index = v_60[0]
    t_repol = t_dat[t_index] 

    APD = t_repol - t_depol


