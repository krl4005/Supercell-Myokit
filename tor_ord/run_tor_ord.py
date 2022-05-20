#%% 
from cProfile import label
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.signal import find_peaks # pip install scipy 

#%% 
#single run
t = time.time()
mod,proto, x = myokit.load('./tor_ord_endo2.mmt')
params = ['i_cal_pca_multiplier', 'i_kr_multiplier', 'i_ks_multiplier', 'i_nal_multiplier', 'i_na_multiplier', 'jup_multiplier', 'i_to_multiplier', 'i_k1_multiplier', 'i_NCX_multiplier', 'i_nak_multiplier', 'i_kb_multiplier']
#mod['multipliers'][params[0]].set_rhs(15)
proto.schedule(5.3, 0.1, 1, 1000, 0)
proto.schedule(0.1, 4, 1000-100, 1000, 1)
sim = myokit.Simulation(mod, proto)
sim.pre(1000*100)
dat = sim.run(1000)

t = dat['engine.time']
v = dat['membrane.v']
c = dat['intracellular_ions.cai']

fig, ax = plt.subplots(1, 1, True, figsize=(12, 8))
ax.plot(t, v, color = "k")
ax.set_xlabel('Time (ms)', fontsize=14)
ax.set_ylabel('Voltage (mV)', fontsize=14)


#%% Calcium transient 
t = time.time()
mod,proto, x = myokit.load('./tor_ord_endo.mmt')
proto.schedule(5.3, 0.1, 1, 1000, 0)
sim = myokit.Simulation(mod, proto)
sim.pre(1000*1000)
dat = sim.run(1000)

t = dat['engine.time']
v = dat['membrane.v']
c = dat['intracellular_ions.cai']
fig, ax = plt.subplots(1, 1, True, figsize=(12, 8))
ax.plot(t, c, color = "k")
ax.set_xlabel('Time (ms)', fontsize=14)
ax.set_ylabel('Intracellular Calcium', fontsize=14)

#%% 
conductances = [1, 1.5, 2, 2.5]

fast_v = []
fast_t = []
slow_v = []
slow_t = []

for conduct in conductances: 
    t = time.time()
    mod,proto, x = myokit.load('./tor_ord_endo.mmt')
    mod['multipliers']['i_cal_pca_multiplier'].set_rhs(conduct)
    proto.schedule(5.3, 1.1, 1, 5000, 0)
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000*5000)
    dat = sim.run(1000)

    slow_v.append(dat['membrane.v'])
    slow_t.append(dat['engine.time'])

    mod2, proto2, x = myokit.load('./tor_ord_endo.mmt')
    proto2.schedule(5.3, 1.1, 1, 500, 0)
    mod2['multipliers']['i_cal_pca_multiplier'].set_rhs(conduct)
    sim2 = myokit.Simulation(mod2, proto2)
    sim2.pre(1000*500)
    dat2 = sim2.run(500)

    fast_v.append(dat2['membrane.v'].tolist())
    fast_t.append(dat2['engine.time'].tolist())

for i in list(range(0,len(fast_t))):
    fast_label = 'fast {}'
    plt.plot(fast_t[i], fast_v[i], label = fast_label.format(i))
    slow_label = 'slow {}'
    plt.plot(slow_t[i], slow_v[i], label = slow_label.format(i))
plt.legend()
plt.xlim(-10,500)
plt.show()



# %%
t = time.time()
cl = 1000
m, p, x = myokit.load('./tor_ord_endo.mmt')
p.schedule(5.3, 10, 0.5, cl, 0)
s = myokit.Simulation(m, p)
s.pre(100*cl)
d = s.run(1*cl)

mod, proto, x = myokit.load('./tor_ord_endo.mmt')
mod['multipliers']['i_cal_pca_multiplier'].set_rhs(3)
mod['multipliers']['i_kr_multiplier'].set_rhs(0.1)
mod['multipliers']['i_ks_multiplier'].set_rhs(1)
mod['multipliers']['i_nal_multiplier'].set_rhs(1)
mod['multipliers']['jup_multiplier'].set_rhs(1)
proto.schedule(5.3, 10, 0.5, cl, 0)
#proto.schedule(0.1, 4014, 1000-100, cl, 1)
sim = myokit.Simulation(mod, proto)
sim.pre(100*cl)

dat = sim.run(1*cl)


vt90 = 0.9*sim.state()[mod.get('membrane.v').indice()]
apd90 = dat.apd(v='membrane.v', threshold = vt90)
print(apd90)

plt.plot(d['engine.time'], d['membrane.v'], color = "black")
plt.plot(dat['engine.time'], dat['membrane.v'], color = "red")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
#plt.axis("off")
IC = sim.state()

#%%
plt.plot(d['engine.time'], d['intracellular_ions.cai'], color = "black")
plt.plot(dat['engine.time'], dat['intracellular_ions.cai'], color = "red")
plt.xlabel("Time (ms)")
plt.ylabel("Intracellular Calcium (mM)")

#%%
plt.plot(d['engine.time'], d['ICaL.ICaL_i'], color = "black")
plt.plot(dat['engine.time'], dat['ICaL.ICaL_i'], color = "red")
plt.xlabel("Time (ms)")
plt.ylabel("ICaL (A/F)")

#%% Proposal Figures:
plt.figure(figsize=(4, 6), dpi=500)
t = time.time()
cl = 1000
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
proto.schedule(5.3, 10, 0.5, cl, 0)
proto.schedule(0.1, 4014, 1000-100, cl, 1)
sim = myokit.Simulation(mod, proto)
sim.pre(100*cl)

dat = sim.run(1*cl)
y_ax = []

mdp = min(dat['membrane.v'])
max_p = max(dat['membrane.v'])
max_p_idx = np.argmax(dat['membrane.v'])
apa = max_p - mdp
dvdt_max = np.max(np.diff(dat['membrane.v'][0:30])/np.diff(dat['engine.time'][0:30]))

for apd_pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    repol_pot = max_p - apa * apd_pct/100
    idx_apd = np.argmin(np.abs(np.array(dat['membrane.v'])[max_p_idx:] - repol_pot))
    apd_val = dat['engine.time'][idx_apd+max_p_idx]
    plt.axhline(y = repol_pot, xmin=0.05, xmax=0.9, color = 'k', alpha = 0.8, linestyle = 'dashed')
    y_ax.append(repol_pot) 
    lab = 'APD' + str(apd_pct)
    plt.annotate(lab, xy =(3.3, 1), xytext =(300, 32-(12*(apd_pct/10))), color = 'black', alpha = 0.8)

#print(y_ax)

plt.annotate('', xy =(0, 30), xytext =(-5, -90), arrowprops = dict(arrowstyle = '<|-|>', color= 'black'))
plt.annotate('DvDt', xy =(3.3, 1), xytext =(-30, -30), rotation = 90, color = 'black', alpha = 0.8)

plt.plot(dat['engine.time'], dat['membrane.v'])
plt.xlim([0,450])
plt.axis('off')

#%% 
plt.figure(figsize=(8, 6), dpi=500)
t = time.time()
cl = 1000
mod, proto, x = myokit.load('./tor_ord_endo.mmt')
proto.schedule(5.3, 10, 0.5, cl, 0)
proto.schedule(0.1, 4014, 1000-100, cl, 1)
sim = myokit.Simulation(mod, proto)
sim.pre(100*cl)

dat = sim.run(1*cl)

y_ax = []
cai = dat['intracellular_ions.cai']

max_cai = np.max(cai)
max_cai_idx = np.argmax(cai)
cat_amp = np.max(cai) - np.min(cai)

for cat_pct in [10, 50, 90]:
    cat_recov = max_cai - cat_amp * cat_pct / 100
    idx_catd = np.argmin(np.abs(cai[max_cai_idx:] - cat_recov))
    catd_val = dat['engine.time'][idx_catd+max_cai_idx]
    plt.axhline(y = cat_recov, xmin=0.05, xmax=0.25+1*(cat_pct/250), color = 'k', alpha = 0.8, linestyle = 'dashed')
    y_ax.append(cat_recov) 

plt.text(180,0.000365,'CaT10', {'ha': 'center', 'va': 'center'}, color = 'black', alpha = 0.8)
plt.text(300,0.000240,'CaT50', {'ha': 'center', 'va': 'center'}, color = 'black', alpha = 0.8)
plt.text(550,0.000115,'CaT90', {'ha': 'center', 'va': 'center'}, color = 'black', alpha = 0.8)
plt.annotate('', xy =(75,0.00038), xytext =(75,0.00007), arrowprops = dict(arrowstyle = '<|-|>', color= 'black'))
plt.text(65,0.00017,'Amplitude', {'ha': 'center', 'va': 'center'}, rotation = 90, color = 'black', alpha = 0.8)

#print(y_ax)

plt.plot(dat['engine.time'], dat['intracellular_ions.cai'])
plt.axis('off')


#%% 

#mod1, proto1, x = myokit.load('./tor_ord_endo.mmt')
proto1.schedule(5.3, 10, 0.5, cl, 0)
#sim = myokit.Simulation(mod1, proto1)
sim.set_state(IC)
dat = sim.run(cl)
plt.plot(dat['engine.time'], dat['membrane.v'])





# %%
plt.plot(dat['engine.time'], dat['IKs.IKs'], label = 'IKs')
plt.plot(dat['engine.time'], dat['ICaL.ICaL'], label = 'ICaL')
plt.plot(dat['engine.time'], dat['INaL.INaL'], label = 'INaL')
plt.plot(dat['engine.time'], dat['IKr.IKr'], label = 'IKr')
plt.legend()

# %%
plt.plot(dat['engine.time'], dat['IKs.IKs'], label = 'IKs')
plt.legend()

# %%
