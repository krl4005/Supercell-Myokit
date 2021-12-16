#%%  INITIAL CONDITIONS
# In order for this to work, make sure the protocol in the mmt file is commented out!!
import myokit
import matplotlib.pyplot as plt
import numpy as np

num = 0     # num represents the model you want to run. 
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
        #p.schedule(1, 1, 1, cl, 0)    # TP06
        p.schedule(5.3, 0, 1, cl, 0) # Tor-Ord 
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

    #p.schedule(1, 1.1, 1, cl, 0)    # TP06
    p.schedule(5.3, 0, 1, cl, 0) # Tor-Ord 
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
cl = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]

base = []
drug = []
drug1 = []
for i in list(range(0,len(cl))):
    print('start', i)
    APD_base = get_APD(cl[i], 0, 0, 1)
    base.append(APD_base)

    APD_drug = get_APD(cl[i], 0, 0, 2.75)   # I_cal * 2.75
    drug.append(APD_drug)

    APD_drug1 = get_APD(cl[i], 0, 1, 0.2)   # I_Kr * 0.2
    drug1.append(APD_drug1)
    print('end', i)

DI = [cl[i]-base[i] for i in list(range(0,len(cl)))]
DI_drug = [cl[i]-drug[i] for i in list(range(0,len(cl)))]
DI_drug1 = [cl[i]-drug1[i] for i in list(range(0,len(cl)))]

plt.plot(DI, base, label='baseline')
plt.plot(DI_drug, drug, label = 'drug - ICaL')
plt.plot(DI_drug1, drug1, label = 'drug - IKr')

plt.xlabel('DI Interval')
plt.ylabel('APD')
plt.legend()
plt.show()

#%% Error Function 2

def error2(base, drug, error):
    #Baseline
    slow_base = base[0]
    fast_base = base[-1] 

    #Drug
    slow_drug = drug[0]
    fast_drug = drug[-1]

    #Percent change calculation
    v_slow = (slow_drug/slow_base)*100
    v_fast = (fast_drug/fast_base)*100
    val = v_slow-v_fast  #error=135 for ICal & error=5000 for IKr

    if error == 1:  #continuous error- the greater the number, the lower the error
        if val<0:
            error = abs(val)*1000

        if val == 0:
            error = 1000

        if val>0:
            error = (1/val)*1000 
    
    if error == 2: #binary error 
        if val >0:
            error = 0

        if val <0:
            error = 1000
    return(error)

# %% Using error function 
i_cal_continuous = error2(base, drug, 1)
i_kr_continuous = error2(base, drug1, 1)
print(i_cal_continuous, i_kr_continuous)

i_cal_binary = error2(base, drug, 2)
i_kr_binary = error2(base, drug1, 2)
print(i_cal_binary, i_kr_binary)
# %%
