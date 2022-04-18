#%% 
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np

#Single Run
mod, proto, x = myokit.load('./kernik.mmt')
proto.schedule(0, 1, 10, 1000, 0)
sim = myokit.Simulation(mod, proto)
dat_normal = sim.run(1400)

t = dat_normal['engine.time']
v = dat_normal['membrane.V']

fig, ax = plt.subplots(1, 1, True, figsize=(12, 8))
ax.plot(t, v, color = "k")
ax.set_xlabel('Time (ms)', fontsize=14)
ax.set_ylabel('Voltage (mV)', fontsize=14)

#%% 
def split_APs(dat):
    v = dat['membrane.V']
    t = dat['engine.time']
    i_stim=np.array(dat['stimulus.i_stim'].tolist())
    AP_S_where = np.where(i_stim == -3.0)[0]
    AP_S_diff = np.where(np.diff(AP_S_where)!=1)[0]+1
    peaks = AP_S_where[AP_S_diff]

    v_APs = np.split(v, peaks)
    t_APs = np.split(t, peaks)
    s = np.split(i_stim, peaks)
    return(v_APs, t_APs)

def calc_APD90(v,t):
        mdp = min(v)
        max_p = max(v)
        max_p_idx = np.argmax(v)
        apa = max_p - mdp
        dvdt_max = np.max(np.diff(v[0:30])/np.diff(t[0:30]))

        repol_pot = max_p - apa * 90/100
        idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
        apd_val = t[idx_apd+max_p_idx]
        return apd_val

#%%
t = time.time()

#baseline
mod, proto, x = myokit.load('./kernik.mmt')
proto.schedule(0, 100, 10, 1000, 0)
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
#simm.pre(100 * 1000) #pre-pace for 100 beats 
dat_mature = simm.run(5000)
#print(f'It took {time.time() - t} s')
tm = dat_mature['engine.time']
vm = dat_mature['membrane.V']

v_APs, t_APs = split_APs(dat_mature)
apd90s = []
for i in list(range(0, len(v_APs))):
    vol = v_APs[i]
    tim = t_APs[i]-1000*i
    apd90 = calc_APD90(vol, tim)
    apd90s.append(apd90-500)
print(apd90s)

fig, ax = plt.subplots(1, 1, True, figsize=(12, 8))
ax.plot(t, v)
ax.plot(tm,vm)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Time (ms)', fontsize=14)
ax.set_ylabel('Voltage (mV)', fontsize=14)
plt.show()


# %%
def get_rrc_error(mod, proto, sim):

    ## RRC CHALLENGE
    proto.schedule(1, 0.2, 5, 1000, 0)
    proto.schedule(0.1, 25, 500, 1000, 1)
    proto.schedule(0.15, 5025, 500, 1000, 1)
    proto.schedule(0.2, 10025, 500, 1000, 1)
    proto.schedule(0.25, 15025, 500, 1000, 1)
    proto.schedule(0.3, 20025, 500, 1000, 1)

    sim = myokit.Simulation(mod, proto)
    sim.pre(100 * 1000) #pre-pace for 100 beats
    dat = sim.run(25000)

    # Pull out APs and find APD90
    v_APs, t_APs = split_APs(dat)
    apd90s = []
    norm_APs = []
    for i in list(range(0, len(v_APs))):
        vol = v_APs[i]
        tim = t_APs[i] -1000*i
        apd90 = calc_APD90(vol, tim)
        apd90s.append(apd90)
        norm_APs.append(apd90)

    ########### APD90 DETECTION ############# 
    stim_idx = [0, 5, 10, 15, 20]
    stim_APs = []
    for i in stim_idx:
        stim_APs.append(apd90s[i])
        norm_APs.remove(apd90s[i])
    
    base_apd90 = np.mean(norm_APs)
    apd_change = []
    for i in list(range(0,len(stim_APs))):
        del_apd = ((stim_APs[i]-base_apd90)/base_apd90)*100
        apd_change.append(del_apd)
            
    #################### RRC DETECTION ###########################
    #global RRC

    RRC_vals = [-0.3, -0.45, -0.6, -0.5, -0.75, -0.9]
    
    for i in list(range(0, len(apd_change))): 
        if apd_change[i] > 40:
            RRC = RRC_vals[i]
            break
        else:
            RRC = -1.0 #if there is no EAD than the stim was not strong enough so error should be zero

    return dat, RRC, apd_change

def get_ind_data(ind):
    mod, proto, x = myokit.load('./kernik.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    proto.schedule(1, 0.1, 1, 1000, 0) #ADDED IN
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats 
    IC = sim.state()

    return mod, proto, sim, IC

#%%
tunable_parameters=['i_cal_pca_multiplier',
                    'i_ks_multiplier',
                    'i_kr_multiplier',
                    'i_na_multiplier',
                    'i_to_multiplier',
                    'i_k1_multiplier',
                    'i_f_multiplier']

initial_params = [1, 
                  1, 
                  1, 
                  187/129, 
                  1,
                  11.24/5.67,
                  1]

keys = [val for val in tunable_parameters]
final = [dict(zip(keys, initial_params))]

print(final[0])

mod, proto, sim, IC = get_ind_data(final)
dat, RRC, apd_change = get_rrc_error(mod, proto, sim)


plt.figure(figsize=[10,3])
plt.plot(dat['engine.time'], dat['membrane.V'])
# %%
