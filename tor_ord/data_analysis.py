#%% 
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from ast import literal_eval
import math
import myokit
from scipy.signal import find_peaks # pip install scipy
from multiprocessing import Pool
import time

print(time.time())

#%% GA 3

def get_ind_data(ind):
    mod, proto, x = myokit.load('./tor_ord_endo2.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto

def get_feature_errors(t,v,cai,i_ion):
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
        apd_val = calc_APD(t,v,apd_pct) 
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

def get_normal_sim_dat(mod, proto):
    """
        Runs simulation for a given individual. If the individuals is None,
        then it will run the baseline model
        Returns
        ------
            t, v, cai, i_ion
    """
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    sim = myokit.Simulation(mod,proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats
    dat = sim.run(5000)
    IC = sim.state()

    # Get t, v, and cai for second to last AP#######################
    #t, v, cai, i_ion = get_last_ap(dat, -2)

    #return (t, v, cai, i_ion, IC)
    return(dat, IC)

def get_last_ap(dat, AP):

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

    return (t, v, cai, i_ion)

def detect_EAD(t, v):
    #find slope
    slopes = []
    for i in list(range(0, len(v)-1)):
        m = (v[i+1]-v[i])/(t[i+1]-t[i])
        slopes.append(round(m, 2))

    #find rises
    pos_slopes = np.where(slopes > np.float64(0.0))[0].tolist()
    pos_slopes_idx = np.where(np.diff(pos_slopes)!=1)[0].tolist()
    pos_slopes_idx.append(len(pos_slopes)) #list must end with last index

    #pull out groups of rises (indexes)
    pos_groups = []
    pos_groups.append(pos_slopes[0:pos_slopes_idx[0]+1])
    for x in list(range(0,len(pos_slopes_idx)-1)):
        g = pos_slopes[pos_slopes_idx[x]+1:pos_slopes_idx[x+1]+1]
        pos_groups.append(g)

    #pull out groups of rises (voltages and times)
    vol_pos = []
    tim_pos = []
    for y in list(range(0,len(pos_groups))):
        vol = []
        tim = []
        for z in pos_groups[y]:
            vol.append(v[z])
            tim.append(t[z])
        vol_pos.append(vol)
        tim_pos.append(tim) 

    #Find EAD given the conditions (voltage>-70 & time>100)
    EADs = []
    EAD_vals = []
    for k in list(range(0, len(vol_pos))):
        if np.mean(vol_pos[k]) > -70 and np.mean(tim_pos[k]) > 100:
            EAD_vals.append(tim_pos[k])
            EAD_vals.append(vol_pos[k])
            EADs.append(max(vol_pos[k])-min(vol_pos[k]))

    #Report EAD 
    if len(EADs)==0:
        info = "no EAD"
        result = 0
    else:
        info = "EAD:", round(max(EADs))
        result = 1
    
    return result

def get_ead_error(ind, code): 

    ## EAD CHALLENGE: Istim = -.1
    if code == 'stim':
        mod, proto = get_ind_data(ind)
        proto.schedule(5.3, 0.1, 1, 1000, 0) 
        proto.schedule(0.1, 3004, 1000-100, 1000, 1) #EAD amp is about 4mV from this
        sim = myokit.Simulation(mod,proto)
        dat = sim.run(5000)
        plt.plot(dat['engine.time'], dat['membrane.v'])

        # Get t, v, and cai for second to last AP#######################
        t, v, cai, i_ion = get_last_ap(dat, -2)

    ## EAD CHALLENGE: ICaL = 15x (acute increase - no prepacing here)
    if code == 'ical':
        mod, proto = get_ind_data(ind)
        mod['multipliers']['i_cal_pca_multiplier'].set_rhs(ind[0]['i_cal_pca_multiplier']*13)
        proto.schedule(5.3, 0.1, 1, 1000, 0) 
        sim = myokit.Simulation(mod,proto)
        dat = sim.run(5000)
        plt.plot(dat['engine.time'], dat['membrane.v'])

        # Get t, v, and cai for second to last AP#######################
        t, v, cai, i_ion = get_last_ap(dat, -2)

    ## EAD CHALLENGE: IKr = 90% block (acute increase - no prepacing here)
    if code == 'ikr':
        mod, proto = get_ind_data(ind)
        mod['multipliers']['i_kr_multiplier'].set_rhs(ind[0]['i_kr_multiplier']*0.05)
        mod['multipliers']['i_kb_multiplier'].set_rhs(ind[0]['i_kb_multiplier']*0.05)
        proto.schedule(5.3, 0.1, 1, 1000, 0) 
        sim = myokit.Simulation(mod,proto)
        dat = sim.run(5000)
        plt.plot(dat['engine.time'], dat['membrane.v'])

        # Get t, v, and cai for second to last AP#######################
        t, v, cai, i_ion = get_last_ap(dat, -2)


    ########### EAD DETECTION ############# 
    EAD = detect_EAD(t,v)

    return t,v,EAD

def detect_RF(t,v):

    #find slopes
    slopes = []
    for i in list(range(0, len(v)-1)):
        m = (v[i+1]-v[i])/(t[i+1]-t[i])
        slopes.append(round(m, 1))

    #find times and voltages at which slope is 0
    zero_slopes = np.where(slopes == np.float64(0.0))[0].tolist()
    zero_slopes_idx = np.where(np.diff(zero_slopes)!=1)[0].tolist()
    zero_slopes_idx.append(len(zero_slopes)) #list must end with last index

    #pull out groups of zero slope (indexes)
    zero_groups = []
    zero_groups.append(zero_slopes[0:zero_slopes_idx[0]+1])
    for x in list(range(0,len(zero_slopes_idx)-1)):
        g = zero_slopes[zero_slopes_idx[x]+1:zero_slopes_idx[x+1]+1]
        zero_groups.append(g)

    #pull out groups of zero slopes (voltages and times)
    vol_pos = []
    tim_pos = []
    for y in list(range(0,len(zero_groups))):
        vol = []
        tim = []
        for z in zero_groups[y]:
            vol.append(v[z])
            tim.append(t[z])
        vol_pos.append(vol)
        tim_pos.append(tim) 


    #Find RF given the conditions (voltage<-70 & time>100)
    no_RF = []
    for k in list(range(0, len(vol_pos))):
        if np.mean(vol_pos[k]) < -70 and np.mean(tim_pos[k]) > 100:
            no_RF.append(tim_pos[k])
            no_RF.append(vol_pos[k])

    #Report EAD 
    if len(no_RF)==0:
        info = "Repolarization failure!"
        result = 1
    else:
        info = "normal repolarization - resting membrane potential from t=", no_RF[0][0], "to t=", no_RF[0][len(no_RF[0])-1]
        result = 0
    return result

def detect_APD(t, v, apd90_base):
    APD90_i = calc_APD(t, v, 90)
    APD90_error = (APD90_i - apd90_base)/(APD90_i)*100
    if APD90_error < 40:
        result_APD = 0
    else:
        result_APD = 1
    return(result_APD)

def calc_APD(t, v, apd_pct):
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    repol_pot = max_p - apa * apd_pct/100
    idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
    apd_val = t[idx_apd+max_p_idx]
    return(apd_val) 

def rrc_search1(IC, ind):
    #Run 6 normal beats and 1 at 0.3 stim to assess RRC 
    all_t = []
    all_v = []
    stims = [0, 0.3]

    mod, proto = get_ind_data(ind)
    mod.set_state(IC) #use state after prepacing
    proto.schedule(5.3, 0.2, 1, 1000, 0)
    proto.schedule(0.3, 5004, 995, 1000, 1)
    sim = myokit.Simulation(mod, proto)
    dat = sim.run(7000)

    t0, v0, cai0, i_ion0 = get_last_ap(dat, 4)
    all_t.append(t0)
    all_v.append(v0)
    result_EAD0 = detect_EAD(t0,v0)
    result_RF0 = detect_RF(t0,v0)

    t3, v3, cai3, i_ion3 = get_last_ap(dat, 5)
    all_t.append(t3)
    all_v.append(v3)
    result_EAD3 = detect_EAD(t3,v3)
    result_RF3 = detect_RF(t3,v3)

    if result_EAD0 == 1 or result_RF0 == 1:
        RRC = 0

    elif result_EAD3 == 0 and result_RF3 == 0:
        # no abnormality at 0.3 stim, return RRC
        RRC = 0.3

    else:
        low = 0.075
        high = 0.3
        while (high-low)>0.0025:
            mid = round((low + (high-low)/2), 4)
            stims.append(mid)

            mod, proto = get_ind_data(ind)
            mod.set_state(IC) #use state after prepacing
            proto.schedule(5.3, 0.2, 1, 1000, 0)
            proto.schedule(mid, 4004, 995, 1000, 1)
            sim = myokit.Simulation(mod, proto)
            dat = sim.run(6000)

            t, v, cai, i_ion = get_last_ap(dat, 4)
            all_t.append(t)
            all_v.append(v)
            result_EAD = detect_EAD(t,v)
            result_RF = detect_RF(t,v)

            if (high-low)<0.0025:
                break 
            
            elif result_EAD == 0 and result_RF == 0:
                # no RA so go from mid to high
                low = mid

            else:
                #repolarization failure so go from mid to low 
                high = mid

        result_EADlast = detect_EAD(all_t[-1],all_v[-1])
        result_RFlast = detect_RF(all_t[-1],all_v[-1])
        if result_EADlast == 0 and result_RFlast == 0:
            RRC = stims[-1] #the last stim attempted should have a repolarization abnormality
        else:
            RRC = stims[-2]

    return(RRC)

def rrc_search(ind, IC):
    all_t = []
    all_v = []
    stims = [0, 0.3]
    APs = list(range(10004, 100004, 5000))

    mod, proto = get_ind_data(ind) 
    proto.schedule(5.3, 0.2, 1, 1000, 0)
    proto.schedule(0.3, 5004, 995, 1000, 1)
    sim = myokit.Simulation(mod, proto)
    sim.set_state(IC)
    dat = sim.run(7000)

    t0, v0, cai0, i_ion0 = get_last_ap(dat, 4)
    all_t.append(t0)
    all_v.append(v0)
    result_EAD0 = detect_EAD(t0,v0)
    result_RF0 = detect_RF(t0,v0)

    t3, v3, cai3, i_ion3 = get_last_ap(dat, 5)
    all_t.append(t3)
    all_v.append(v3)
    result_EAD3 = detect_EAD(t3,v3)
    result_RF3 = detect_RF(t3,v3)

    if result_EAD0 == 1 or result_RF0 == 1:
        RRC = 0

    elif result_EAD3 == 0 and result_RF3 == 0:
        # no abnormality at 0.3 stim, return RRC
        RRC = 0.3

    else:
        low = 0.075
        high = 0.3
        EADs = []
        RFs = []
        for i in list(range(0,len(APs))):
            mid = round((low + (high-low)/2), 4)
            stims.append(mid)

            sim.reset()
            sim.set_state(IC)
            proto.schedule(mid, APs[i], 995, 1000, 1)
            sim.set_protocol(proto)
            dat = sim.run(APs[i]+2000)

            t, v, cai, i_ion = get_last_ap(dat, int((APs[i]-4)/1000))
            all_t.append(t)
            all_v.append(v)
            result_EAD = detect_EAD(t,v)
            EADs.append(result_EAD)
            result_RF = detect_RF(t,v)
            RFs.append(result_RF)

            if (high-low)<0.0025:
                break 
            
            elif result_EAD == 0 and result_RF == 0:
                # no RA so go from mid to high
                low = mid

            else:
                #repolarization failure so go from mid to low 
                high = mid
        
        for i in list(range(1, len(EADs))):
            if EADs[-i] == 0 and RFs[-i] == 0:
                RRC = stims[-i] 
            else:
                RRC = 0 #in this case there would be no stim without an RA

    return(RRC, all_t, all_v, stims)

def assess_challenges(ind):

    ## EAD CHALLENGE: Istim = -.1
    mod, proto = get_ind_data(ind)
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    proto.schedule(0.1, 4004, 1000-100, 1000, 1) #EAD amp is about 4mV from this
    sim = myokit.Simulation(mod,proto)
    sim.pre(100*1000)
    dat = sim.run(6000)
    IC = sim.state()

    ## EAD CHALLENGE: ICaL = 30x (acute increase - no prepacing here)
    #sim.reset()
    sim.set_state(IC)
    sim.set_constant('multipliers.i_cal_pca_multiplier', ind[0]['i_cal_pca_multiplier']*30)
    dat1 = sim.run(2000)

    ## EAD CHALLENGE: IKr = 80% block (acute increase - no prepacing here)
    #sim.reset()
    sim.set_state(IC)
    sim.set_constant('multipliers.i_cal_pca_multiplier', ind[0]['i_cal_pca_multiplier'])
    sim.set_constant('multipliers.i_kr_multiplier', ind[0]['i_kr_multiplier']*0.05)
    sim.set_constant('multipliers.i_kb_multiplier', ind[0]['i_kb_multiplier']*0.05)
    dat2 = sim.run(2000)

    return dat, dat1, dat2

#%% READ IN DATA

#path = 'c:\\Users\\Kristin\\Desktop\\iter4\\g50_p200_e2\\trial1'
error_thres = 2000

#pop = pd.read_csv(path + '\\pop.csv')
#error = pd.read_csv(path + '\\error.csv')
pop = pd.read_csv('pop.csv')
error = pd.read_csv('error.csv')

#%% GROUP BEST INIDIDUALS FOR ALL GENERATIONS
best_error = []
best_ind = []

for gen in list(range(1, len(error.columns))):
    for ind in list(range(0, len(error[error.columns[gen]]))):
        if  error[error.columns[gen]][ind] <= error_thres:
            best_error.append(error[error.columns[gen]][ind])
            best_ind.append(literal_eval(pop[error.columns[gen]][ind]))

print("ind length", len(best_ind))

label = ['GCaL', 'GKs', 'GKr', 'GNaL', 'GNa', 'Gto', 'GK1', 'GNCX', 'GNaK', 'Gkb']
df_cond = pd.DataFrame(best_ind, columns=label)  
df_cond.to_csv('best_conds.csv', index=False)

df_error = pd.DataFrame(best_error, columns=['error'])  
df_error.to_csv('best_error.csv', index=False)


#%%
def eval_challenges(ind):
    tunable_parameters=['i_cal_pca_multiplier', 'i_ks_multiplier', 'i_kr_multiplier', 'i_nal_multiplier', 'i_na_multiplier', 'i_to_multiplier', 'i_k1_multiplier', 'i_NCX_multiplier', 'i_nak_multiplier', 'i_kb_multiplier']
    base = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    baseline = [dict(zip(tunable_parameters, base))]
    opt = best_ind[ind]
    optimized = [dict(zip(tunable_parameters, opt))]

        
    #dat, dat1, dat2 = assess_challenges(baseline)
    dat, dat1, dat2 = assess_challenges(optimized)


    #plt.plot(dat['engine.time'], dat['membrane.v'], label = 'assess alternans and i_stim = -0.1')
    #plt.plot(dat1['engine.time'], dat1['membrane.v'], label = 'ICaL = 30x')
    #plt.plot(dat2['engine.time'], dat2['membrane.v'], label = 'Assess immunity to RF')
    #plt.legend()

    #alternans
    overall_result = []
    t_alt1, v_alt1, cai_alt1, i_ion_alt1 = get_last_ap(dat, -5)
    t_alt2, v_alt2, cai_alt2, i_ion_alt2 = get_last_ap(dat, -4)
    apd901 = calc_APD(t_alt1, v_alt1, 90)
    apd902 = calc_APD(t_alt2, v_alt2, 90)
    if np.abs(apd902-apd901)<1:
        overall_result.append(0)
    else:
        overall_result.append(1)

    #stim challenge
    t_stim, v_stim, cai_stim, i_ion_stim = get_last_ap(dat, -2)
    stim_resultEAD = detect_EAD(t_stim, v_stim)
    stim_resultRF = detect_RF(t_stim, v_stim) 
    if stim_resultEAD == 0 and stim_resultRF == 0:
        overall_result.append(0)
    else:
        overall_result.append(1)

    #ical challenge
    t_ical, v_ical, cai_ical, i_ion_ical = get_last_ap(dat1, -2) 
    ical_resultEAD = detect_EAD(t_ical, v_ical)
    ical_resultRF = detect_RF(t_ical, v_ical) 
    if ical_resultEAD == 0 and ical_resultRF == 0:
        overall_result.append(0)
    else:
        overall_result.append(1)

    #ikr challenge
    t_rf, v_rf, cai_rf, i_ion_rf = get_last_ap(dat2, -2) 
    rf_resultEAD = detect_EAD(t_rf, v_rf)
    rf_resultRF = detect_RF(t_rf, v_rf) 
    if rf_resultEAD == 0 and rf_resultRF == 0:
        overall_result.append(0)
    else:
        overall_result.append(1)

    return(overall_result)

#to use on local
#results = []
#for i in list(range(0, len(best_ind))):
#    print(i)
#    overall_result=eval_challenges1(i)
#    results.append(overall_result)

# to use multithreding on cluster
if __name__ == "__main__":
    p = Pool()
    results = p.map(eval_challenges, range(0, len(best_ind)))
    

df_challenges = pd.DataFrame(results, columns = ['alternans', 'i_stim = 0.1', 'iCaL = 30x', 'RF'])  
df_challenges.to_csv('challenges.csv', index=False)

print(time.time())
#%% 
######################################################################################
### BELOW ARE FUNCTIONS THAT ARE NO LONGER NEEDED BUT WANT TO KEEP FOR BOOKEEPING ###
######################################################################################
#%% CALCULATE EXACT RRC FOR BINARY GA

#def calc_rrc(ind):
#    tunable_parameters=['i_cal_pca_multiplier', 'i_ks_multiplier', 'i_kr_multiplier', 'i_nal_multiplier', 'i_na_multiplier', 'i_to_multiplier', 'i_k1_multiplier', 'i_NCX_multiplier', 'i_nak_multiplier', 'i_kb_multiplier']
#    opt = best_ind[ind]
#    optimized = [dict(zip(tunable_parameters, opt))]

#    m, p = get_ind_data(optimized)
#    dat, IC = get_normal_sim_dat(m, p) 
#    RRC = rrc_search(IC, optimized)
#    return(RRC)

## to use on local
#all_RRCs = []
#for i in list(range(0, len(best_ind))):
#for i in list(range(0, 10)):
#    RRC = calc_rrc(ind)
#    all_RRCs.append(RRC) 

# to use multithreding on cluster
#if __name__ == "__main__":
#    p = Pool()
#    all_RRCs = p.map(calc_rrc, range(0, len(best_ind)))
#    print("RRCs", all_RRCs)

#df_rrc = pd.DataFrame(all_RRCs, columns = ['RRC'])  
#df_rrc.to_csv('RRCs.csv', index=False)

#%% ENSURE EACH INDIVIDUAL HAS A NORMAL AMOUNT OF BEAT-BEAT VARIABILITY (NO ALTERNANS)
#def calc_alternans(ind):
#    tunable_parameters=['i_cal_pca_multiplier', 'i_ks_multiplier', 'i_kr_multiplier', 'i_nal_multiplier', 'i_na_multiplier', 'i_to_multiplier', 'i_k1_multiplier', 'i_NCX_multiplier', 'i_nak_multiplier', 'i_kb_multiplier']
#    opt = best_ind[ind]
#    optimized = [dict(zip(tunable_parameters, opt))]

#    m, p = get_ind_data(optimized)
#    dat, IC = get_normal_sim_dat(m, p) 

#    apd90s = []
#    for i in list(range(0,4)):
#        t,v,cai,i_ion = get_last_ap(dat, i)
#        apd90 = calc_APD(t,v,90)
#        apd90s.append(apd90)

#    return(apd90s)

## to use on local
#check_alternans = []
#for i in list(range(0, len(best_error1))):
#    apd90s = calc_alternans(i)
#    check_alternans.append(apd90s)

# to use multithreding on cluster
#if __name__ == "__main__":
#    p = Pool()
#    check_alternans = p.map(calc_alternans, range(0, len(best_ind)))
#    print("potential alternans:", check_alternans) 

#df_alternans = pd.DataFrame(check_alternans, columns = ['AP 1', 'AP 2', 'AP 3', 'AP 4'])  
#df_alternans.to_csv('alternans.csv', index=False)

#%% RUN CHALLENGES FOR ALL IN LIST OF BEST INDIVIDUALS & ELIMINATE INDS THAT WERENT IMMUNE TO ALL CHALLENGES
#tunable_parameters=['i_cal_pca_multiplier', 'i_ks_multiplier', 'i_kr_multiplier', 'i_nal_multiplier', 'i_na_multiplier', 'i_to_multiplier', 'i_k1_multiplier', 'i_NCX_multiplier', 'i_nak_multiplier', 'i_kb_multiplier']
#base = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#baseline = [dict(zip(tunable_parameters, base))]
#opt = best_ind[ind]
#opt = [0.17459823241839148, 1.5948018015979026, 1.9971728343561777, 1.7375355635092948, 0.5289193832021285, 0.6569060762063939, 1.9433298888741581,  0.11342645756459585, 1.2902123791965632, 1.8861099266439645]
#optimized = [dict(zip(tunable_parameters, opt))]

#def eval_challenges(ind):
    #tunable_parameters=['i_cal_pca_multiplier', 'i_ks_multiplier', 'i_kr_multiplier', 'i_nal_multiplier', 'i_na_multiplier', 'i_to_multiplier', 'i_k1_multiplier', 'i_NCX_multiplier', 'i_nak_multiplier', 'i_kb_multiplier']
    #opt = best_ind[ind]
    #optimized = [dict(zip(tunable_parameters, opt))]

#    overall_result = []

    # Challenge - stimulus
#    stim_t1, stim_v1, stim_EAD1 = get_ead_error(ind, "stim")
#    stim_resultEAD = detect_EAD(stim_t1, stim_v1)
#    stim_resultRF = detect_RF(stim_t1, stim_v1) 
#    if stim_resultEAD == 0 and stim_resultRF == 0:
#        overall_result.append(0)
#    else:
#        overall_result.append(1)

    # Challenge - ICaL
#    ical_t1, ical_v1, ical_EAD1 = get_ead_error(ind, "ical")
#    ical_resultEAD = detect_EAD(ical_t1, ical_v1)
#    ical_resultRF = detect_RF(ical_t1, ical_v1) 
#    if ical_resultEAD == 0 and ical_resultRF == 0:
#        overall_result.append(0)
#    else:
#        overall_result.append(1)

    # Challenge - IKr
#    ikr_t1, ikr_v1, ikr_EAD1 = get_ead_error(ind, "ikr")
#    ikr_resultEAD = detect_EAD(ikr_t1, ikr_v1)
#    ikr_resultRF = detect_RF(ikr_t1, ikr_v1) 
#    if ikr_resultEAD == 0 and ikr_resultRF == 0:
#        overall_result.append(0)
#    else:
#        overall_result.append(1)
    
#    return(overall_result)

## to use on local
#challenges = []
#for i in list(range(0, len(best_error))):
#    overall_result = eval_challenges(baseline)
#    challenges.append(overall_result)
#print(challenges)

# to use multithreding on cluster
#if __name__ == "__main__":
#    p = Pool()
#    challenges = p.map(eval_challenges, range(0, len(best_ind)))
#    print("Challenge Answers:", challenges)

#df_challenges = pd.DataFrame(challenges, columns = ['Stimulus Challenge', 'ICal Challenge', 'IKr challenge'])  
#df_challenges.to_csv('challenges.csv', index=False)