# %%
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from ast import literal_eval

#individuals = pickle.load(open("individuals", "rb"))

pop = pd.read_csv('pop.csv')
last_gen = []

for i in list(range(0, len(pop["gen24"]))):
    last_gen.append(literal_eval(pop["gen24"][i]))

i_cal = []
i_ks = []
i_kr = []
i_nal = []
jup = []

for i in list(range(0,len(last_gen))):
    i_cal.append(last_gen[i][0])
    i_ks.append(last_gen[i][1]) 
    i_kr.append(last_gen[i][2])
    i_nal.append(last_gen[i][3])  
    jup.append(last_gen[i][4]) 



# %%
plt.scatter([1]*len(i_cal),i_cal)
plt.scatter([2]*len(i_ks),i_ks)
plt.scatter([3]*len(i_kr),i_kr)
plt.scatter([4]*len(i_nal),i_nal)
plt.scatter([5]*len(jup),jup)
plt.show()


# %%
