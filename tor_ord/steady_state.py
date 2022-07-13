#%% 
from cProfile import label
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.signal import find_peaks # pip install scipy 
import pandas as pd

def get_feature_errors(t, v, cai):
    """
    Compares the simulation data for an individual to the baseline Tor-ORd values. The returned error value is a sum of the differences between the individual and baseline values.
    Returns
    ------
        error
    """

    ap_features = {}

    # Returns really large error value if cell AP is not valid 
    if ((min(v) > -60) or (max(v) < 0)):
        return 50000000 

    # Voltage/APD features#######################
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    dvdt_max = np.max(np.diff(v[0:30])/np.diff(t[0:30]))

    ap_features['Vm_peak'] = max_p
    #ap_features['Vm_t'] = t[max_p_idx]
    ap_features['dvdt_max'] = dvdt_max

    for apd_pct in [40, 50, 90]:
        repol_pot = max_p - apa * apd_pct/100
        idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
        apd_val = t[idx_apd+max_p_idx]
        
        ap_features[f'apd{apd_pct}'] = apd_val
 
    ap_features['triangulation'] = ap_features['apd90'] - ap_features['apd40']
    ap_features['RMP'] = np.mean(v[len(v)-50:len(v)])

    # Calcium/CaT features######################## 
    max_cai = np.max(cai)
    max_cai_idx = np.argmax(cai)
    max_cai_time = t[max_cai_idx]
    cat_amp = np.max(cai) - np.min(cai)
    ap_features['cat_amp'] = cat_amp * 1e5 #added in multiplier since number is so small
    ap_features['cat_peak'] = max_cai_time

    for cat_pct in [90]:
        cat_recov = max_cai - cat_amp * cat_pct / 100
        idx_catd = np.argmin(np.abs(cai[max_cai_idx:] - cat_recov))
        catd_val = t[idx_catd+max_cai_idx]

        ap_features[f'cat{cat_pct}'] = catd_val 

    return ap_features

def get_ap(dat, AP):

    # Get t, v, and cai for second to last AP#######################
    i_stim = dat['stimulus.i_stim']

    # This is here so that stim for EAD doesnt interfere with getting the whole AP
    for i in list(range(0,len(i_stim))):
        if abs(i_stim[i])<50:
            i_stim[i]=0

    peaks = find_peaks(-np.array(i_stim), distance=100)[0]
    start_ap = peaks[AP] #TODO change start_ap to be after stim, not during
    end_ap = peaks[AP+1]

    t = np.array(dat['engine.time'][start_ap:end_ap])
    t = t - t[0]
    max_idx = np.argmin(np.abs(t-995))
    t = t[0:max_idx]
    end_ap = start_ap + max_idx

    v = np.array(dat['membrane.v'][start_ap:end_ap])
    cai = np.array(dat['intracellular_ions.cai'][start_ap:end_ap])
    i_ion = np.array(dat['membrane.i_ion'][start_ap:end_ap])

    return (t, v, cai)

#%% 
from ast import literal_eval
path = 'c:\\Users\\Kristin\\Desktop\\Christini Lab\\Research Data\\supercell-myokit\\cluster\\fit+RRC\\iter2\\g10_p200_e2\\trial2'
#pop = pd.read_csv(path + '\\pop.csv')
pop = pd.read_csv('pop.csv')

pop_col = pop.columns.tolist()
first_gen = []

for i in list(range(0, len(pop[pop_col[1]]))):
    first_gen.append(literal_eval(pop[pop_col[1]][i]))

#%% Without Prepacing
mods = []
for i in list(range(0,len(first_gen[0:10]))):
    run_time=6000
    mod,proto, x = myokit.load('./tor_ord_endo2.mmt')
    vals = first_gen[i]
    mod['multipliers']['i_cal_pca_multiplier'].set_rhs(vals[0])
    mod['multipliers']['i_ks_multiplier'].set_rhs(vals[1])
    mod['multipliers']['i_kr_multiplier'].set_rhs(vals[2])
    mod['multipliers']['i_nal_multiplier'].set_rhs(vals[3])
    mod['multipliers']['i_na_multiplier'].set_rhs(vals[4])
    mod['multipliers']['i_to_multiplier'].set_rhs(vals[5])
    mod['multipliers']['i_k1_multiplier'].set_rhs(vals[6])
    mod['multipliers']['i_NCX_multiplier'].set_rhs(vals[7])
    mod['multipliers']['i_nak_multiplier'].set_rhs(vals[8])
    mod['multipliers']['i_kb_multiplier'].set_rhs(vals[9])
    proto.schedule(5.3, 0.1, 1, 1000, 0)
    sim = myokit.Simulation(mod, proto)
    dat = sim.run(run_time)

    t = dat['engine.time']
    v = dat['membrane.v']

    #plt.figure(1)
    #plt.plot(t,v)

    #plt.figure(2)
    features = []
    end = (run_time/1000)-1
    for i in list(range(0, int(end))):
        t_AP, v_AP, cai_AP = get_ap(dat, i)
        ap_features = get_feature_errors(t_AP, v_AP, cai_AP)
        #features.append(ap_features)
        features.append(round(ap_features['apd90']))
        #lab = 'AP'+str(i)+': APD90 - '+str(round(ap_features['apd90']))
        #plt.plot(t_AP,v_AP, label = lab)
    mods.append(features)

    #plt.title("No Prepacing")
    #plt.legend()
    #plt.show()
    

# %% With Prepacing 
mods_100pre = []
for i in list(range(0,len(first_gen[0:10]))):
    vals = first_gen[i]
    run_time=6000
    mod,proto, x = myokit.load('./tor_ord_endo2.mmt')
    mod['multipliers']['i_cal_pca_multiplier'].set_rhs(vals[0])
    mod['multipliers']['i_ks_multiplier'].set_rhs(vals[1])
    mod['multipliers']['i_kr_multiplier'].set_rhs(vals[2])
    mod['multipliers']['i_nal_multiplier'].set_rhs(vals[3])
    mod['multipliers']['i_na_multiplier'].set_rhs(vals[4])
    mod['multipliers']['i_to_multiplier'].set_rhs(vals[5])
    mod['multipliers']['i_k1_multiplier'].set_rhs(vals[6])
    mod['multipliers']['i_NCX_multiplier'].set_rhs(vals[7])
    mod['multipliers']['i_nak_multiplier'].set_rhs(vals[8])
    mod['multipliers']['i_kb_multiplier'].set_rhs(vals[9])
    proto.schedule(5.3, 0.1, 1, 1000, 0)
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000*100)
    dat = sim.run(run_time)

    t = dat['engine.time']
    v = dat['membrane.v']

    #plt.figure(3)
    #plt.plot(t,v)

    #plt.figure(4)
    features = []
    end = (run_time/1000)-1
    for i in list(range(0, int(end))):
        t_AP, v_AP, cai_AP = get_ap(dat, i)
        ap_features = get_feature_errors(t_AP, v_AP, cai_AP)
        #features.append(ap_features)
        features.append(round(ap_features['apd90']))
        #lab = 'AP'+str(i)+': APD90 - '+str(round(ap_features['apd90']))
        #plt.plot(t_AP,v_AP, label = lab)
    mods_100pre.append(features)

    #plt.legend()
    #plt.title("Prepacing 100 beats")
    #plt.show()

# %%
mods_1000pre = []
for i in list(range(0,len(first_gen[0:10]))):
    vals = first_gen[i]
    run_time=6000
    mod,proto, x = myokit.load('./tor_ord_endo2.mmt')
    mod['multipliers']['i_cal_pca_multiplier'].set_rhs(vals[0])
    mod['multipliers']['i_ks_multiplier'].set_rhs(vals[1])
    mod['multipliers']['i_kr_multiplier'].set_rhs(vals[2])
    mod['multipliers']['i_nal_multiplier'].set_rhs(vals[3])
    mod['multipliers']['i_na_multiplier'].set_rhs(vals[4])
    mod['multipliers']['i_to_multiplier'].set_rhs(vals[5])
    mod['multipliers']['i_k1_multiplier'].set_rhs(vals[6])
    mod['multipliers']['i_NCX_multiplier'].set_rhs(vals[7])
    mod['multipliers']['i_nak_multiplier'].set_rhs(vals[8])
    mod['multipliers']['i_kb_multiplier'].set_rhs(vals[9])
    proto.schedule(5.3, 0.1, 1, 1000, 0)
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000*100)
    dat = sim.run(run_time)

    t = dat['engine.time']
    v = dat['membrane.v']

    #plt.figure(3)
    #plt.plot(t,v)

    #plt.figure(4)
    features = []
    end = (run_time/1000)-1
    for i in list(range(0, int(end))):
        t_AP, v_AP, cai_AP = get_ap(dat, i)
        ap_features = get_feature_errors(t_AP, v_AP, cai_AP)
        #features.append(ap_features)
        features.append(round(ap_features['apd90']))
        #lab = 'AP'+str(i)+': APD90 - '+str(round(ap_features['apd90']))
        #plt.plot(t_AP,v_AP, label = lab)
    mods_1000pre.append(features)

    #plt.legend()
    #plt.title("Prepacing 1000 beats")
    #plt.show()
# %%

df_0 = pd.DataFrame(mods)
df_0.to_csv('c:\\Users\\Kristin\\Desktop\\mods.csv')

df_100 = pd.DataFrame(mods_100pre)
df_100.to_csv('c:\\Users\\Kristin\\Desktop\\mods_100pre.csv')

df_1000 = pd.DataFrame(mods_1000pre)
df_1000.to_csv('c:\\Users\\Kristin\\Desktop\\mods_1000pre.csv')


# %% TESTING PREPACING 
#single run
run_time = 5000

tunable_parameters = ['i_cal_pca_multiplier', 'i_kr_multiplier', 'i_ks_multiplier', 'i_nal_multiplier', 'i_na_multiplier', 'jup_multiplier', 'i_to_multiplier', 'i_k1_multiplier', 'i_NCX_multiplier', 'i_nak_multiplier', 'i_kb_multiplier']
initial_params = first_gen[0]
keys = [val for val in tunable_parameters]
ind = [dict(zip(keys, initial_params))]

mod, proto, x = myokit.load('./tor_ord_endo2.mmt')
if ind is not None:
    for k, v in ind[0].items():
        mod['multipliers'][k].set_rhs(v)

proto.schedule(5.3, 0.1, 1, 1000, 0)
#proto.schedule(0.1, 4, 1000-100, 1000, 1)
sim = myokit.Simulation(mod, proto)
sim.pre(1000*100)
dat = sim.run(run_time)
ic = sim.state()

t = dat['engine.time']
v = dat['membrane.v']
cai = dat['intracellular_ions.cai']
plt.plot(t,v) 

features = []
end = (run_time/1000)-1
for i in list(range(0, int(end))):
    t_AP, v_AP, cai_AP = get_ap(dat, i)
    ap_features = get_feature_errors(t_AP, v_AP, cai_AP)
    features.append(round(ap_features['apd90']))

print(features)


#%%
mod1, proto1, x = myokit.load('./tor_ord_endo2.mmt')
if ind is not None:
    for k, v in ind[0].items():
        mod1['multipliers'][k].set_rhs(v)

mod.set_state(ic)
proto.schedule(5.3, 0.1, 1, 1000, 0)
#proto.schedule(0.1, 4, 1000-100, 1000, 1)
sim1 = myokit.Simulation(mod, proto)
dat1 = sim1.run(5000)

t1 = dat1['engine.time']
v1 = dat1['membrane.v']
cai1 = dat1['intracellular_ions.cai']

plt.plot(t1,v1)

features1 = []
end = (run_time/1000)-1
for i in list(range(0, int(end))):
    t1_AP, v1_AP, cai1_AP = get_ap(dat1, i)
    ap_features = get_feature_errors(t1_AP, v1_AP, cai1_AP)
    features1.append(round(ap_features['apd90']))

print(features1)
# %%
