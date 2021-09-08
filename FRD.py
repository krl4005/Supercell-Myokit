
#%% FUNCTIONS
from re import M
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas 

def evaluate_APD90(dat): 
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

        repol_pot = max_p - (apa * 90/100)
        idx_apd = np.argmin(np.abs(AP[n][max_p_idx:] - repol_pot))
        apd_val = t1[n][idx_apd+max_p_idx]-t1[n][0]
        APD.insert(n, apd_val) 

    return(APD)


def pacing_protocol(cl, drug_amount, drug_name):
    
    t0 = time.time()
    APD_purturb = []
    t_data = []
    v_data = []

    mod, proto, x = myokit.load('./tor_ord_endo.mmt')

    #reset
    mod['multipliers']['i_cal_pca_multiplier'].set_rhs(1)
    mod['multipliers']['i_kr_multiplier'].set_rhs(1)
    mod['multipliers']['i_nal_multiplier'].set_rhs(1)
    mod['multipliers']['i_ks_multiplier'].set_rhs(1)
    mod['multipliers']['jup_multiplier'].set_rhs(1)
    mod['multipliers']['i_to_multiplier'].set_rhs(1)

    proto.schedule(5.3, 0.1, 1, cl, 0)
    sim0 = myokit.Simulation(mod, proto)
    sim0.pre(1000 * 1000) #pre-pace for 100 beats 
    dat0 = sim0.run(cl)
    IC = sim0.state()

    for i in list(range(0, len(drug_amount))):
        mod.set_state(IC)
        mod['multipliers'][drug_name].set_rhs(drug_amount[i])
        sim = myokit.Simulation(mod, proto)
        sim.pre(100 * 1000)
        dat = sim.run(cl)

        #add data to list to plot 
        t = dat['engine.time'].tolist()
        t_data.insert(i, t)
        v = dat['membrane.v'].tolist()
        v_data.insert(i, v)

        APD_val = evaluate_APD90(dat)
        APD_purturb.insert(i, APD_val)

    return(APD_purturb, t_data, v_data) 


def normalize_apd(APD_data):
    APD_norm = []
    for i in list(range(0,len(APD_data))):

        #APD
        x = min(APD_data)[0]
        new_val = (APD_data[i][0]/x) * 100
        APD_norm.insert(i,new_val)

    return(APD_norm)

def normalize_conduct(conductances):
    conduct_norm = []
    for i in list(range(0,len(conductances))):
        new_val = conductances[i]*100
        conduct_norm.insert(i, new_val)

    return(conduct_norm)

print('cell complete')

#%% ANALYSIS - ICaL
slow_cl = 5000
fast_cl = 500
normal_cl = 1000

ical = 'i_cal_pca_multiplier'
ical_amount = [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
APD_ical_fast, ical_t_data_fast, ical_v_data_fast = pacing_protocol(fast_cl, ical_amount, ical)
APD_ical_slow, ical_t_data_slow, ical_v_data_slow = pacing_protocol(slow_cl, ical_amount, ical)

norm_APD_ical_fast = normalize_apd(APD_ical_fast)
norm_APD_ical_slow = normalize_apd(APD_ical_slow)
norm_conduct_ical= normalize_conduct(ical_amount)

fig, axs = plt.subplots(3, constrained_layout=True, figsize=(15,15))
axs[0].plot(norm_conduct_ical, norm_APD_ical_slow, label="slow")
axs[0].plot(norm_conduct_ical, norm_APD_ical_fast, label = "fast")
axs[0].legend()

for i in list(range(0, len(ical_amount))):
    lab_fast = "i_cal_multiplier = {}, APD= {}".format(ical_amount[i], APD_ical_fast[i][0])
    axs[1].plot(ical_t_data_fast[i], ical_v_data_fast[i], label = lab_fast)
    axs[1].set_title('Fast Pacing: CL = 500', fontsize=16)
    axs[1].legend()

    lab_slow = "i_cal_multiplier = {}, APD= {}".format(ical_amount[i], APD_ical_slow[i][0])
    axs[2].plot(ical_t_data_slow[i], ical_v_data_slow[i], label = lab_slow)
    axs[2].set_title('Slow Pacing: CL = 5000', fontsize=16)
    axs[2].legend()

plt.xlim([-20, 500])
plt.show()

#%% PLOTS - IKs 
iks = 'i_ks_multiplier'
iks_amount = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
APD_iks_fast, iks_t_data_fast, iks_v_data_fast = pacing_protocol(fast_cl, iks_amount, iks)
APD_iks_slow, iks_t_data_slow, iks_v_data_slow = pacing_protocol(slow_cl, iks_amount, iks)

norm_APD_iks_fast = normalize_apd(APD_iks_fast)
norm_APD_iks_slow = normalize_apd(APD_iks_slow)
norm_conduct_iks= normalize_conduct(iks_amount)

fig, axs = plt.subplots(3, constrained_layout=True, figsize=(15,15))
axs[0].plot(norm_conduct_iks, norm_APD_iks_slow, label="slow")
axs[0].plot(norm_conduct_iks, norm_APD_iks_fast, label = "fast")
axs[0].legend()
axs[0].invert_xaxis()

for i in list(range(0, len(iks_amount))):
    lab_fast = "i_ks_multiplier = {}, APD= {}".format(iks_amount[i], APD_iks_fast[i][0])
    axs[1].plot(iks_t_data_fast[i], iks_v_data_fast[i], label = lab_fast)
    axs[1].set_title('Fast Pacing: CL = 500', fontsize=16)
    axs[1].legend()

    lab_slow = "i_ks_multiplier = {}, APD= {}".format(iks_amount[i], APD_iks_slow[i][0])
    axs[2].plot(iks_t_data_slow[i], iks_v_data_slow[i], label = lab_slow)
    axs[2].set_title('Slow Pacing: CL = 5000', fontsize=16)
    axs[2].legend()

plt.xlim([-20, 500])
plt.show()

#%% PLOTS - IKr
ikr = 'i_kr_multiplier'
ikr_amount = [1.0, 0.8, 0.6, 0.4, 0.2]
APD_ikr_fast, ikr_t_data_fast, ikr_v_data_fast = pacing_protocol(fast_cl, ikr_amount, ikr)
APD_ikr_slow, ikr_t_data_slow, ikr_v_data_slow = pacing_protocol(slow_cl, ikr_amount, ikr)

norm_APD_ikr_fast = normalize_apd(APD_ikr_fast)
norm_APD_ikr_slow = normalize_apd(APD_ikr_slow)
norm_conduct_ikr= normalize_conduct(ikr_amount)

fig, axs = plt.subplots(3, constrained_layout=True, figsize=(15,15))
axs[0].plot(norm_conduct_ikr, norm_APD_ikr_slow, label="slow")
axs[0].plot(norm_conduct_ikr, norm_APD_ikr_fast, label = "fast")
axs[0].legend()
axs[0].invert_xaxis()

for i in list(range(0, len(ikr_amount))):
    lab_fast = "i_kr_multiplier = {}, APD= {}".format(ikr_amount[i], APD_ikr_fast[i][0])
    axs[1].plot(ikr_t_data_fast[i], ikr_v_data_fast[i], label = lab_fast)
    axs[1].set_title('Fast Pacing: CL = 500', fontsize=16)
    axs[1].legend()

    lab_slow = "i_kr_multiplier = {}, APD= {}".format(ikr_amount[i], APD_ikr_slow[i][0])
    axs[2].plot(ikr_t_data_slow[i], ikr_v_data_slow[i], label = lab_slow)
    axs[2].set_title('Slow Pacing: CL = 5000', fontsize=16)
    axs[2].legend()

plt.xlim([-20, 800])
plt.show()



# %% Error function 1:

# An ideal class III agent would instead prolong APs in a forward rate dependent (FRD) manner. 
# That is, it would prolong the APD at fast heart rates but induce minimal prolongation at slow heart rates. 

def error1(APD_fast, APD_slow):
    #Calculate Ratios
    fast = APD_fast[len(APD_fast)-1][0] - APD_fast[0][0] #31
    slow = APD_slow[len(APD_slow)-1][0] - APD_slow[0][0] #39

    #Compare
    if 0 > slow > 20:
        slow_error = 0
    else:
        slow_error = 2500

    if 20 < fast:
        fast_error = 0
    else:
        fast_error = 2500

    if slow < fast:
        add_error = 0
    else:
        add_error = 2500

    #Calculate Error
    FRD_error = slow_error + fast_error + add_error
    return(FRD_error)

FDR_error1_ical = error1(APD_ical_fast, APD_ical_slow)
FDR_error1_iks = error1(APD_iks_fast, APD_iks_slow)
FDR_error1_ikr = error1(APD_ikr_fast, APD_ikr_slow)

print(FDR_error1_ical, FDR_error1_iks, FDR_error1_ikr)

# %% Error Function 2:
# if last normalized APD for fast > last normalized APD for slow than error = 0

def error2(APD_norm_fast, APD_norm_slow):
    #Calculate Ratios
    fast = APD_norm_fast[len(APD_norm_fast)-1]
    slow = APD_norm_slow[len(APD_norm_slow)-1]

    if abs(fast-slow)>1:
        #Compare
        if fast > slow:
            error = 0
        else:
            error = 5000
    else:
        error = 0 #this would be for a neutral situation...should this be zero?

    #Calculate Error
    FRD_error = error
    return(FRD_error)

FDR_error2_ical = error2(norm_APD_ical_fast, norm_APD_ical_slow)
FDR_error2_iks = error2(norm_APD_iks_fast, norm_APD_iks_slow)
FDR_error2_ikr = error2(norm_APD_ikr_fast, norm_APD_ikr_slow)

print(FDR_error2_ical, FDR_error2_iks, FDR_error2_ikr)


# %%
