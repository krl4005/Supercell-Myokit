#%%  INITIAL CONDITIONS
import myokit
import matplotlib.pyplot as plt
import numpy as np

num = 1     # num represents the model you want to run. 
            # Tor-Ord = 0 
            # TP06 = 1

param = 0   #param represents the conductance variable you want to analyze.
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
        p.schedule(1, 1.1, 1, cl, 0)    # TP06
        #p.schedule(5.3, 1.1, 1, cl, 1) # Tor-Ord 
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
            if d < 1:
                v_60.append(i)
            else:
                wrong.append(i)

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
    values = [[1, 1.25, 1.5, 2, 2.5, 2.75], [1, 0.8, 0.6, 0.4, 0.2], [1, 0.8, 0.6, 0.4, 0.2]]
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

# %% APD Restitution PLOTS 
APD_fast = get_APD(500, 1, 0, 1)
APD_norm = get_APD(1000, 1, 0, 1)
APD_slow = get_APD(5000, 1, 0, 1)

print(APD_fast, APD_norm, APD_slow)


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

FDR_error1_ical = error1(APD_fast, APD_slow)
FDR_error1_iks = error1(APD_fast, APD_slow)
FDR_error1_ikr = error1(APD_fast, APD_slow)

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

FDR_error2_ical = error2(norm_APD_fast, norm_APD_slow)
FDR_error2_iks = error2(norm_APD_fast, norm_APD_slow)
FDR_error2_ikr = error2(norm_APD_fast, norm_APD_slow)

print(FDR_error2_ical, FDR_error2_iks, FDR_error2_ikr)

# %%
