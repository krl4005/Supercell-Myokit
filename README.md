# ToR-ORd Supercell

### Convert Tor-ORd CellML to .mmt and run in Python

The Tor-ORd model has already been converted and is saved as `tor_ord_endo.mmt`. You can follow the steps below if you're interested in learning how to convert CellML files to .mmt.

1. If you don't have it already, install [Myokit](http://myokit.org/install)
2. Download the [Tor-ORd model](https://models.physiomeproject.org/e/5f1/Tomek_model_endo.cellml/view) from CellML – the file you'll need is already in this repo
3. Run `convert_to_mmt.py` to convert `Tomek_model_endo.cellml` to a `.mmt` file.

### Update the .mmt file to align with Myokit best practices

You should only complete this step if you converted the Tor-ORd file from CellML to .mmt.

Add the following code to the bottom of the `tor_ord.mmt` file:
```
[[protocol]]
#Level  Start    Length   Period   Multiplier
4.0      10       1        1000      0
```

Add the following on lines 49-60. If you search for the multiplier variables in `tor_ord_endo.mmt`, you'll see how I used them to scale current.

```
[engine]
time = 0 [ms]
    in [ms]
    bind time
pace = 0
    bind pace

[stimulus]
i_stim = engine.pace * amplitude
    in [A/F]
amplitude = -10 [A/F]
    in [A/F]

[multipliers]
i_cal_pca_multiplier = 1
i_kr_multiplier = 1
i_ks_multiplier = 1
i_nal_multiplier = 1
jup_multiplier = 1
```

Remove the following from the `[environment]` component:
```
time = 0 [ms] bind time
    in [ms]
    oxmeta: time
```

Remove the following from the `[membrane]` component:
```
Istim = piecewise(environment.time >= i_Stim_Start and environment.time <= i_Stim_End and environment.time - i_Stim_Start - floor((environment.time - i_Stim_Start) / i_Stim_Period) * i_Stim_Period <= i_Stim_PulseDuration, i_Stim_Amplitude, 0 [A/F])
    in [A/F]
    oxmeta: membrane_stimulus_current
i_Stim_Amplitude = -53 [A/F]
    in [A/F]
    oxmeta: membrane_stimulus_current_amplitude
i_Stim_End = 1e17 [ms]
    in [ms]
i_Stim_Period = 1000 [ms]
    in [ms]
    oxmeta: membrane_stimulus_current_period
i_Stim_PulseDuration = 1 [ms]
    in [ms]
    oxmeta: membrane_stimulus_current_duration
i_Stim_Start = 0 [ms]
    in [ms]
    oxmeta: membrane_stimulus_current_offset
```

Remove `membrane.Istim` from the `dot(ki)` line.

In the `[membrane]` `dot(v)` equation, change `Istim` to `stimulus.i_stim`

Add the following to the `[membrane]` component:
```
i_ion = INa.INa + INaL.INaL + Ito.Ito + ICaL.ICaL + ICaL.ICaNa + ICaL.ICaK + IKr.IKr + IKs.IKs + IK1.IK1 + INaCa.INaCa_i + INaCa.INaCa_ss + INaK.INaK + INab.INab + IKb.IKb + IpCa.IpCa + ICab.ICab + ICl.IClCa + ICl.IClb + I_katp.I_katp
```

### Run and plot the Tor-ORd model

Run `run_tor_ord.py` to visualize the `Tor-ORd` model.


### Run the Genetic Algorithm

The `supercell_ga.py` file includes a GA designed to create a supercell. Below is a list of the important components.

- The following code is used to set variables that are used to configure the GA. You can use this code to change things like, population size, generation, etc.
```
feature_targets = {'dvdt_max': [80, 86, 92],
                   'apd10': [2, 8, 20],
                   'apd50': [200, 220, 250],
                   'apd90': [250, 270, 300],
                   'cat_amp': [3E-4, 3.5E-4, 4E-4],
                   'cat10': [80, 100, 120],
                   'cat50': [200, 220, 240],
                   'cat90': [450, 470, 490]}

# 1. Initializing GA hyperparameters
global GA_CONFIG
GA_CONFIG = Ga_Config(population_size=3, 
                      max_generations=2,
                      params_lower_bound=0.5,
                      params_upper_bound=2,
                      tunable_parameters=['i_cal_pca_multiplier',
                                          'i_ks_multiplier',
                                          'i_kr_multiplier',
                                          'i_nal_multiplier',
                                          'jup_multiplier'],
                      mate_probability=0.9,
                      mutate_probability=0.9,
                      gene_swap_probability=0.2,
                      gene_mutation_probability=0.2,
                      tournament_size=4,
                      cost='function_2',
                      feature_targets=feature_targets)

```

- The following two lines of code are all DEAP code. These lines define the problem as a minimization, and create the `Individual` class that is used to track parameters and fitness values during the GA. This [DEAP tutorial](https://deap.readthedocs.io/en/master/overview.html) explains these lines in greater detail.

```py
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

creator.create('Individual', list, fitness=creator.FitnessMin)
```

- The `start_ga()` function call at the bottom of the file start the GA.
- Inside `start_ga()`, you create a `toolbox` object, which is another class from DEAP. In the `start_ga()`, you register a bunch of functions (e.g. `_initialize_individuals()`, `_evaluate_fitness()`, `_mate()`, and `_mutate()`), so the DEAP toolbox knows how to initialize the population, evaluate individuals, mate, and mutate.
- The `run_ga()` line of code will start the GA.
- Follow the comments inside `run_ga()` to see how the code works.
- If you want to adjust your cost function, open `_evaluate_fitness`. In this function, you can define the types of simulation protocols and fitnesses you want to track, and then return the fitness value/error. Currently, `feature_error()` is used to run 50 s of AP simulations, and then calculate the error in AP features. The `ead_error()` function is incomplete, but will be used to measure the size of EADs after increasing the `I_CaL` current to 8x baseline.
- The `start_ga()` function returns the conductances and fitnesses for every individual from all generations – this array is saved to `final_population` and printed.


















