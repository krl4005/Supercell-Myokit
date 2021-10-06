#%%  INITIAL CONDITIONS
# In order for this to work, make sure the protocol in the mmt file is commented out!!
import myokit
import matplotlib.pyplot as plt
import numpy as np

num = 1     # num represents the model you want to run. 
            # Tor-Ord = 0 
            # TP06 = 1

param = 1   #param represents the conductance variable you want to analyze.
            # i_caL = 0
            # i_kr = 1
            # i_ks = 2



#%% FUNCTIONS
def FRD(cl, num, param):

    # SET UP
    conductances, norm_conduct = get_conduct(param) 
    drug, label = get_drug(param) 

    models = ['./tor_ord_endo.mmt','./TP06.mmt']
    vol = ['membrane.v','membrane.V']
    t0 = ['engine.time','environment.time']

    v_dat = []
    t_dat = []
    apd = []


    for conduct in conductances:
        # Get model and protocol, create simulation
        m, p, x = myokit.load(models[num])
        m['multipliers'][drug].set_rhs(conduct)
        p.schedule(1, 1, 1, cl, 0)    # TP06
        #p.schedule(5.3, 1.1, 1, cl, 0) # Tor-Ord 
        s = myokit.Simulation(m, p)
        print(conduct)
        s.pre(1000*cl)

        # Run simulation
        d = s.run(cl)

        # Calculate APD
        v = d[vol[num]].tolist()
        t = d[t0[num]].tolist()
        depol_index = v.index(max(v[0:150]))
        t_depol = t[depol_index]
        v_60 = []
        wrong = []
        for i in list(range(depol_index, len(v))):
            d = np.abs(-60-v[i])
            if d < 2:
                print(d)
                v_60.append(i)
            else:
                wrong.append(i)

        print(v_60)
        repol_index = v_60[0]
        t_repol = t[repol_index] 

        APD = t_repol - t_depol
        print(APD)
        #vt90 = 0.9*s.state()[m.get(vol[num]).indice()]
        #apd90 = d.apd(v=vol[num], threshold = vt90)['duration']
        #print(apd90[0])

        # Add data from this ical simulation to list 
        v_dat.append(v)
        t_dat.append(t)
        #apd.append(apd90)
        apd.append(APD)
    
    return(v_dat, t_dat, apd)

def get_drug(param):
    drugs = ['i_cal_pca_multiplier', 'i_kr_multiplier', 'i_ks_multiplier']
    drug = drugs[param]
    label = drug + '= {}, APD={}'
    return(drug, label) 


def normalize_apd(APD_data):
    APD_norm = []
    for i in list(range(0,len(APD_data))):

        #APD
        control = APD_data[0]
        new_val = (APD_data[i]/control) * 100
        APD_norm.append(new_val)
        
    return(APD_norm)


def get_conduct(param):
    values = [[1, 1.5, 2, 2.5, 2.75], [1, 0.8, 0.6, 0.4, 0.2], [1, 0.8, 0.6, 0.4, 0.2]]
    conductances = values[param]

    conduct_norm = []
    for i in list(range(0,len(conductances))):
        new_val = conductances[i]*100
        conduct_norm.append(new_val)

    return(conductances, conduct_norm)

def get_APD(cl, num, param, conduct):

    models = ['./tor_ord_endo.mmt','./TP06.mmt']
    vol = ['membrane.v','membrane.V']

    # Get model and protocol, create simulation
    m, p, x = myokit.load(models[num])

    if conduct != 1:
        drug, label = get_drug(param)
        m['multipliers'][drug].set_rhs(conduct)

    p.schedule(1, 1.1, 1, cl, 0)    # TP06
    #p.schedule(5.3, 1.1, 1, cl, 1) # Tor-Ord 
    s = myokit.Simulation(m, p)
    s.pre(1000*cl)

    # Run simulation
    d = s.run(cl)

    vt90 = 0.9*s.state()[m.get(vol[num]).indice()]
    apd90 = d.apd(v=vol[num], threshold = vt90)['duration']
    
    return(apd90[0])

#%% RUN FUNCTIONS 

v_fast, t_fast, APD_fast = FRD(500, num, param)
v_slow, t_slow, APD_slow = FRD(5000, num, param)
norm_APD_fast = normalize_apd(APD_fast)
norm_APD_slow = normalize_apd(APD_slow)
drug, label = get_drug(param) 
conductances, norm_conduct= get_conduct(param)


# %% PLOTS from Paper 
fig, axs = plt.subplots(3, constrained_layout=True, figsize=(15,15))
axs[0].plot(norm_conduct, norm_APD_slow, label="slow")
axs[0].plot(norm_conduct, norm_APD_fast, label = "fast")
axs[0].legend()
for i in list(range(0, len(conductances))):
    lab_fast = label.format(conductances[i], APD_fast[i])
    axs[1].plot(t_fast[i], v_fast[i], label = lab_fast)
    axs[1].set_title('Fast Pacing: CL = 500', fontsize=16)
    axs[1].legend()
    #axs[1].set_xlim(-10,490)

    lab_slow = label.format(conductances[i], APD_slow[i])
    axs[2].plot(t_slow[i], v_slow[i], label = lab_slow)
    axs[2].set_title('Slow Pacing: CL = 5000', fontsize=16)
    axs[2].legend()
    axs[2].set_xlim(-10,600)
plt.show()


# %% Error function 1:
# This error function is based on the conductance vs %APD plot from the paper.
# If the fast APD is greater than the slow APD the error is zero because the FRD is evident. 

def error1(APD_norm_fast, APD_norm_slow):
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

FDR_error1_ical = error1(norm_APD_fast, norm_APD_slow)
FDR_error1_iks = error1(norm_APD_fast, norm_APD_slow)
FDR_error1_ikr = error1(norm_APD_fast, norm_APD_slow)

print(FDR_error1_ical, FDR_error1_iks, FDR_error1_ikr)

# %% APD Restitution PLOTS 
APD_fast = get_APD(500, 1, 0, 1)
APD_mid = get_APD(2250, 1, 0, 1)
APD_slow = get_APD(5000, 1, 0, 1)
plt.plot([500-APD_fast, 2250-APD_mid, 5000-APD_slow], [APD_fast, APD_mid, APD_slow], label = 'baseline')

APD_fast_ical = get_APD(500, 1, 0, 2.75)   # I_cal * 2.75
APD_mid_ical = get_APD(2250, 1, 0, 2.75)   # I_cal * 2.75
APD_slow_ical = get_APD(5000, 1, 0, 2.75)  # I_cal * 2.75
plt.plot([500-APD_fast_ical, 2250-APD_mid_ical, 5000-APD_slow_ical], [APD_fast_ical, APD_mid_ical, APD_slow_ical], label = 'FRD')

APD_fast_ikr = get_APD(500, 1, 1, 0.2)   # I_kr * 0.2
APD_mid_ikr = get_APD(2250, 1, 1, 0.2)
APD_slow_ikr = get_APD(5000, 1, 1, 0.2)  # I_kr * 0.2
plt.plot([500-APD_fast_ikr, 2250-APD_mid_ikr, 5000-APD_slow_ikr], [APD_fast_ikr, APD_mid_ikr, APD_slow_ikr], label = 'RRD')

plt.xlabel('DI Interval')
plt.ylabel('APD')
plt.legend()
plt.show

# %% Scaled APD restitution curve 
scale_ical_fast = APD_fast_ical-(APD_mid_ical-APD_mid)
scale_ical_mid = APD_mid_ical-(APD_mid_ical-APD_mid)
scale_ical_slow = APD_slow_ical-(APD_mid_ical-APD_mid)

scale_ikr_fast = APD_fast_ikr-(APD_mid_ikr-APD_mid)
scale_ikr_mid = APD_mid_ikr-(APD_mid_ikr-APD_mid)
scale_ikr_slow = APD_slow_ikr-(APD_mid_ikr-APD_mid)

plt.plot([500-APD_fast, 2250-APD_mid, 5000-APD_slow], [APD_fast, APD_mid, APD_slow], label = 'baseline')
plt.plot([500-APD_fast_ical, 2250-APD_mid_ical, 5000-APD_slow_ical], [scale_ical_fast, scale_ical_mid, scale_ical_slow], label = 'FRD')
plt.plot([500-APD_fast_ikr, 2250-APD_mid_ikr, 5000-APD_slow_ikr], [scale_ikr_fast, scale_ikr_mid, scale_ikr_slow], label = 'RRD')
plt.xlabel('DI Interval')
plt.ylabel('Scaled APD')
plt.legend()
plt.show

#%% 
def error2(num, param, conduct):

    # BASELINE 
    APD_fast = get_APD(500, num, 0, 1)
    APD_mid = get_APD(2250, num, 0, 1)
    APD_slow = get_APD(5000, num, 0, 1)
    print(APD_fast, APD_mid, APD_slow)

    # DRUG FOR FRD ANALYSIS
    APD_fast_drug = get_APD(500, num, param, conduct)   
    APD_mid_drug = get_APD(2250, num, param, conduct)  
    APD_slow_drug = get_APD(5000, num, param, conduct)
    print(APD_fast_drug, APD_mid_drug, APD_slow_drug)  

    # NORMALIZE
    norm = APD_mid_drug-APD_mid
    scale_drug_fast = APD_fast_drug - norm
    scale_drug_slow = APD_slow_drug - norm

    error_fast = (abs(1/(scale_drug_fast-APD_fast)))*10000  #the closer to baseline at fast, the higher the error
    error_slow = (abs(scale_drug_slow-APD_slow))*100        #the closer to baseline, the lower the error 
    error2 = error_slow + error_fast 
    return(error2) 

error2_FRD = error2(1, 0, 2.75)   #I_CaL = 2.75 // error = 864
error2_RRD = error2(1, 1, 0.2)    #I_Kr = 0.2  //  error = 1865
print(error2_FRD, error2_RRD)
# %%
