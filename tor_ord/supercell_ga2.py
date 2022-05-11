"""Runs a genetic algorithm for parameter tuning to develop a Super cell.
"""
#%%
import seaborn as sns 
import random
from math import log10
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # pip install scipy
from multiprocessing import Pool
import numpy as np
import pandas as pd

from deap import base, creator, tools # pip install deap
import myokit
import pickle
import time

class Ga_Config():
    def __init__(self,
             population_size,
             max_generations,
             params_lower_bound,
             params_upper_bound,
             iks_lower_bound,
             iks_upper_bound,
             tunable_parameters,
             mate_probability,
             mutate_probability,
             gene_swap_probability,
             gene_mutation_probability,
             tournament_size,
             cost,
             feature_targets):
        self.population_size = population_size
        self.max_generations = max_generations
        self.params_lower_bound = params_lower_bound
        self.params_upper_bound = params_upper_bound
        self.iks_lower_bound = iks_lower_bound
        self.iks_upper_bound = iks_upper_bound
        self.tunable_parameters = tunable_parameters
        self.mate_probability = mate_probability
        self.mutate_probability = mutate_probability
        self.gene_swap_probability = gene_swap_probability
        self.gene_mutation_probability = gene_mutation_probability
        self.tournament_size = tournament_size
        self.cost = cost
        self.feature_targets = feature_targets

def run_ga(toolbox):
    """
    Runs an instance of the genetic algorithm.
    Returns
    -------
        final_population : List[Individuals]
    """
    print('Evaluating initial population.')

    # 3. Calls _initialize_individuals and returns initial population
    population = toolbox.population(GA_CONFIG.population_size)


    # 4. Calls _evaluate_fitness on every individual in the population
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit,)
    # Note: visualize individual fitnesses with: population[0].fitness
    gen_fitnesses = [ind.fitness.values[0] for ind in population]

    print(f'\tAvg fitness is: {np.mean(gen_fitnesses)}')
    print(f'\tBest fitness is {np.min(gen_fitnesses)}')

    # Store initial population details for result processing.
    final_population = [population]
    df_pop = pd.DataFrame()
    df_fit = pd.DataFrame() 

    for generation in range(1, GA_CONFIG.max_generations):
        print('Generation {}'.format(generation))
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        # Offspring are chosen through tournament selection. They are then
        # cloned, because they will be modified in-place later on.

        # 5. DEAP selects the individuals 
        selected_offspring = toolbox.select(population, len(population))

        offspring = [toolbox.clone(i) for i in selected_offspring]

        # 6. Mate the individualse by calling _mate()
        for i_one, i_two in zip(offspring[::2], offspring[1::2]):
            if random.random() < GA_CONFIG.mate_probability:
                toolbox.mate(i_one, i_two)
                del i_one.fitness.values
                del i_two.fitness.values

        # 7. Mutate the individualse by calling _mutate()
        for i in offspring:
            if random.random() < GA_CONFIG.mutate_probability:
                toolbox.mutate(i)
                del i.fitness.values

        # All individuals who were updated, either through crossover or
        # mutation, will be re-evaluated.

        # 8. Evaluating the offspring of the current generation
        updated_individuals = [i for i in offspring if not i.fitness.values]
        fitnesses = toolbox.map(toolbox.evaluate, updated_individuals)
        for ind, fit in zip(updated_individuals, fitnesses):
            ind.fitness.values = (fit,)

        population = offspring

        gen_fitnesses = [ind.fitness.values[0] for ind in population]

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        print(f'\tAvg fitness is: {np.mean(gen_fitnesses)}')
        print(f'\tBest fitness is {np.min(gen_fitnesses)}')

        final_population.append(population)
        
        #Save pop and gen as ga loops 
        label = 'gen'+ str(generation)  
        df_fit[label] = gen_fitnesses
        df_fit.to_csv('error.csv', index=False)

        pop_list = []
        for i in list(range(0,len(population))):
            x = list(population[i][0].values())
            pop_list.append(x)
        df_pop[label] = pop_list
        df_pop.to_csv('pop.csv', index=False)

    return final_population

def _initialize_individuals():
    """
    Creates the initial population of individuals. The initial 
    population 
    Returns:
        An Individual with conductance parameters 
    """
    # Builds a list of parameters using random upper and lower bounds.
    lower_exp = log10(GA_CONFIG.params_lower_bound)
    upper_exp = log10(GA_CONFIG.params_upper_bound)

    iks_lower_exp = log10(GA_CONFIG.iks_lower_bound)
    iks_upper_exp = log10(GA_CONFIG.iks_upper_bound)
    #initial_params = [10**random.uniform(lower_exp, upper_exp)
    #                  for i in range(0, len(
    #                      GA_CONFIG.tunable_parameters))]

    initial_params = []
    for i in range(0, len(GA_CONFIG.tunable_parameters)):
        if i == 1: #increase bounds for iks to try to allow for better convergence 
            initial_params.append(10**random.uniform(iks_lower_exp, iks_upper_exp))
        else:
            initial_params.append(10**random.uniform(lower_exp, upper_exp))

    keys = [val for val in GA_CONFIG.tunable_parameters]
    return dict(zip(keys, initial_params))

def _mate(i_one, i_two):
    """Performs crossover between two individuals.
    There may be a possibility no parameters are swapped. This probability
    is controlled by `GA_CONFIG.gene_swap_probability`. Modifies
    both individuals in-place.
    Args:
        i_one: An individual in a population.
        i_two: Another individual in the population.
    """
    for key, val in i_one[0].items():
        if random.random() < GA_CONFIG.gene_swap_probability:
            i_one[0][key],\
                i_two[0][key] = (
                    i_two[0][key],
                    i_one[0][key])

def _mutate(individual):
    """Performs a mutation on an individual in the population.
    Chooses random parameter values from the normal distribution centered
    around each of the original parameter values. Modifies individual
    in-place.
    Args:
        individual: An individual to be mutated.
    """
    keys = [k for k, v in individual[0].items()]

    for key in keys:
        if key == 'i_ks_multiplier':
            if random.random() < GA_CONFIG.gene_mutation_probability:
                new_param = -1

                while ((new_param < GA_CONFIG.iks_lower_bound) or
                    (new_param > GA_CONFIG.iks_upper_bound)):
                    new_param = np.random.normal(
                            individual[0][key],
                            individual[0][key] * .1)

                individual[0][key] = new_param
        else:
            if random.random() < GA_CONFIG.gene_mutation_probability:
                new_param = -1

                while ((new_param < GA_CONFIG.params_lower_bound) or
                    (new_param > GA_CONFIG.params_upper_bound)):
                    new_param = np.random.normal(
                            individual[0][key],
                            individual[0][key] * .1)

                individual[0][key] = new_param

def get_ind_data(ind):
    mod, proto, x = myokit.load('./tor_ord_endo2.mmt')
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    proto.schedule(5.3, 0.1, 1, 1000, 0) #ADDED IN
    sim = myokit.Simulation(mod, proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats 
    IC = sim.state()

    return mod, proto, sim, IC

def _evaluate_fitness(ind):

    mod, proto, sim, IC = get_ind_data(ind)

    feature_error = get_feature_errors(sim)
    if feature_error == 50000000:
        return feature_error

    #mod.set_state(IC)
    #ead_fitness = get_ead_error(mod, proto, sim, ind) 
    mod.set_state(IC)
    rrc_fitness = get_rrc_error(mod, proto, sim)

    fitness = feature_error + rrc_fitness

    return fitness

def get_feature_errors(sim):
    """
    Compares the simulation data for an individual to the baseline Tor-ORd values. The returned error value is a sum of the differences between the individual and baseline values.
    Returns
    ------
        error
    """
    t,v,cai,i_ion = get_normal_sim_dat(sim)

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
    ap_features['Vm_t'] = t[max_p_idx]
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
    cat_amp = np.max(cai) - np.min(cai)
    #ap_features['cat_amp'] = cat_amp * 1e5 #added in multiplier since number is so small

    for cat_pct in [10, 50, 90]:
        cat_recov = max_cai - cat_amp * cat_pct / 100
        idx_catd = np.argmin(np.abs(cai[max_cai_idx:] - cat_recov))
        catd_val = t[idx_catd+max_cai_idx]

    #    ap_features[f'cat{cat_pct}'] = catd_val 

    error = 0

    if GA_CONFIG.cost == 'function_1':
        for k, v in ap_features.items():
            error += (GA_CONFIG.feature_targets[k][1] - v)**2
    else:
        for k, v in ap_features.items():
            if ((v < GA_CONFIG.feature_targets[k][0]) or
                    (v > GA_CONFIG.feature_targets[k][2])):
                error += 1000 

    return error

def get_normal_sim_dat(sim):
    """
        Runs simulation for a given individual. If the individuals is None,
        then it will run the baseline model
        Returns
        ------
            t, v, cai, i_ion
    """

    #sim.pre(1000 * 100) #pre-pace for 100 beats
    dat = sim.run(5000)

    # Get t, v, and cai for second to last AP#######################
    t, v, cai, i_ion = get_last_ap(dat, -2)

    return (t, v, cai, i_ion)

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

def get_ead_error(mod, proto, sim, ind): 
    ## EAD CHALLENGE: ICaL = 15x (acute increase - no prepacing here)
    #mod['multipliers']['i_cal_pca_multiplier'].set_rhs(ind[0]['i_cal_pca_multiplier']*15)

    ## EAD CHALLENGE: Istim = -.1
    proto.schedule(0.1, 3004, 1000-100, 1000, 1) #EAD amp is about 4mV from this
    sim = myokit.Simulation(mod, proto)
    dat = sim.run(5000)

    t,v,cai,i_ion = get_last_ap(dat, -2)

    ########### EAD DETECTION ############# 
    EAD = detect_EAD(t,v)

    #################### ERROR CALCULATION #######################
    error = 0

    if GA_CONFIG.cost == 'function_1':
        error += (0 - (1000*EAD))**2
    else:
        error += 1000*EAD #Since the baseline EAD is 4mV this is multipled by 1000 to get on the same scale as RRC error

    return error

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

def get_rrc_error(mod, proto, sim):

    ## RRC CHALLENGE
    stims = [0, 0.025, 0.05, 0.075, 0.1, 0.125]
    proto.schedule(5.3, 0.2, 1, 1000, 0)
    proto.schedule(stims[0], 4, 995, 1000, 1)
    proto.schedule(stims[1], 5004, 995, 1000, 1)
    proto.schedule(stims[2], 10004, 995, 1000, 1)
    proto.schedule(stims[3], 15004, 995, 1000, 1)
    proto.schedule(stims[4], 20004, 995, 1000, 1)
    proto.schedule(stims[5], 25004, 995, 1000, 1)

    sim = myokit.Simulation(mod, proto)
    sim.pre(1000 * 1000) #pre-pace for 1000 beats
    dat = sim.run(25000)

    # Pull out APs with RRC stimulus 
    vals = []
    for i in [0, 5, 10, 15, 20]:
        t, v, cai, i_ion = get_last_ap(dat, i)
        plt.plot(t, v)

        ########### EAD DETECTION ############# 
        result_EAD = detect_EAD(t,v) 

        ########### RF DETECTION ############# 
        result_RF = detect_RF(t,v)

        # if EAD and RF place 0 in val list 
        # 0 indicates no RF or EAD for that RRC challenge
        if result_EAD == 0 and result_RF == 0:
            vals.append(0)
        else:
            vals.append(1)

    #################### RRC DETECTION & ERROR CALCULATION ###########################
    #global RRC
    #global E_RRC

    #find first y in list --> that will represent the RRC
    pos_error = [2500, 2000, 1500, 1000, 500, 0]
    for v in list(range(0, len(vals))): 
        if vals[v] == 1:
            RRC = -stims[v-1] #RRC will be the value before the first RF or EAD
            E_RRC = pos_error[v-1]
            break
        else:
            RRC = -1.25 #if there is no EAD or RF than the stim was not strong enough so error should be zero
            E_RRC = 0


    #################### ERROR CALCULATION #######################
    error = 0

    if GA_CONFIG.cost == 'function_1':
        error += (0 - (E_RRC))**2
    else:
        error += E_RRC

    return error

def plot_generation(inds,
                    gen=None,
                    is_top_ten=True,
                    lower_bound=.1,
                    upper_bound=10):
    if gen is None:
        gen = len(inds) - 1

    pop = inds[gen]

    pop.sort(key=lambda x: x.fitness.values[0])
    best_ind = pop[0]

    if is_top_ten:
        pop = pop[0:10]

    keys = [k for k in pop[0][0].keys()]
    empty_arrs = [[] for i in range(len(keys))]
    all_ind_dict = dict(zip(keys, empty_arrs))

    fitnesses = []

    for ind in pop:
        for k, v in ind[0].items():
            all_ind_dict[k].append(v)

        fitnesses.append(ind.fitness.values[0])

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    curr_x = 0

    for k, conds in all_ind_dict.items():
        for i, g in enumerate(conds):
            g = log10(g)
            x = curr_x + np.random.normal(0, .01)
            g_val = 1 - fitnesses[i] / max(fitnesses)
            axs[0].scatter(x, g, color=(0, g_val, 0))
            #if i < 10:
            #    axs[0].scatter(x-.1, g, color='r')


        curr_x += 1


    curr_x = 0

    axs[0].hlines(0, -.5, (len(keys)-.5), colors='grey', linestyle='--')
    axs[0].set_xticks([i for i in range(0, len(keys))])
    axs[0].set_xticklabels(['GCaL', 'GKs', 'GKr', 'GNaL', 'GNa', 'Gto', 'GK1', 'GNCX', 'Gnak', 'Gkb'], fontsize=10)
    axs[0].set_ylim(log10(lower_bound), 
                    log10(upper_bound))
    axs[0].set_ylabel('Log10 Conductance', fontsize=14)

    mod, proto, sim, IC = get_ind_data(best_ind)
    t, v, cai, i_ion = get_normal_sim_dat(sim)
    axs[1].plot(t, v, 'b--', label='Best Fit')
    

    mod, proto, sim, IC = get_ind_data(None)
    t, v, cai, i_ion = get_normal_sim_dat(sim)
    axs[1].plot(t, v, 'k', label='Original Tor-ORd')

    axs[1].set_ylabel('Voltage (mV)', fontsize=14)
    axs[1].set_xlabel('Time (ms)', fontsize=14)

    axs[1].legend()

    fig.suptitle(f'Generation {gen+1}', fontsize=14)

    plt.show()
    plt.savefig("GA_output.png")

def start_ga(pop_size=5, max_generations=5):
    feature_targets = {'dvdt_max': [300, 347, 355],
                       'apd40': [85, 198, 324],
                       'apd50': [106, 220, 350],
                       'apd90': [178, 271, 443],
                       'Vm_peak': [7.3, 33, 40],
                       'Vm_t': [0, 1.6, 14],
                       'triangulation': [62, 73, 153],
                       'RMP': [-94, -88, -78]}

    # 1. Initializing GA hyperparameters
    global GA_CONFIG
    GA_CONFIG = Ga_Config(population_size=pop_size,
                          max_generations=max_generations,
                          params_lower_bound=0.1,
                          params_upper_bound=2,
                          iks_lower_bound = 0.1,
                          iks_upper_bound = 2,
                          tunable_parameters=['i_cal_pca_multiplier',
                                              'i_ks_multiplier',
                                              'i_kr_multiplier',
                                              'i_nal_multiplier',
                                              'i_na_multiplier',
                                              'i_to_multiplier',
                                              'i_k1_multiplier',
                                              'i_NCX_multiplier',
                                              'i_nak_multiplier',
                                              'i_kb_multiplier'],
                          mate_probability=0.9,
                          mutate_probability=0.9,
                          gene_swap_probability=0.2,
                          gene_mutation_probability=0.2,
                          tournament_size=2,
                          cost='function_2',
                          feature_targets=feature_targets)

    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

    creator.create('Individual', list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register('init_param',
                     _initialize_individuals)
    toolbox.register('individual',
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.init_param,
                     n=1)
    toolbox.register('population',
                     tools.initRepeat,
                     list,
                     toolbox.individual)

    toolbox.register('evaluate', _evaluate_fitness)
    toolbox.register('select',
                     tools.selTournament,
                     tournsize=GA_CONFIG.tournament_size)
    toolbox.register('mate', _mate)
    toolbox.register('mutate', _mutate)

    # To speed things up with multi-threading
    #p = Pool()
    #toolbox.register("map", p.map)
    #toolbox.register("map", map)

    # Use this if you don't want multi-threading
    toolbox.register("map", map)

    # 2. Calling the GA to run
    final_population = run_ga(toolbox)

    return final_population

# Final population includes list of individuals from each generation
# To access an individual from last gen:
# final_population[-1][0].fitness.values[0] Gives you fitness/error
# final_population[-1][0][0] Gives you dictionary with conductance values

def main():
    all_individuals = start_ga(pop_size=5, max_generations=5)
    plot_generation(all_individuals, gen=None, is_top_ten=False)
    return(all_individuals)

if __name__=='__main__':
    all_individuals = main()

#%% 
# Put all errors into a list 
dimen = np.shape(all_individuals)
gen = dimen[0]
pop = dimen[1]

error = []
for g in list(range(0,gen)):

    gen_error = []

    for i in list(range(0,pop)):
        e = all_individuals[g][i].fitness.values[0]
        gen_error.append(e)

    error.append(gen_error)

error_df = pd.DataFrame()

for g in list(range(0,gen)):
    label = 'gen'+ str(g) 
    error_df[label] = error[g]

error_df.to_csv('error.csv', index=False)

# save individuals as pickle 
#pickle.dump(all_individuals, open( "individuals", "wb" ) )

# %%
