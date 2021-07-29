# ToR-ORd Supercell

### Convert Tor-ORd CellML to .mmt and run in Python

This step has already been completed, but you can complete it if you would like to see how CellML -> .mmt works.

1. If you don't have it already, install [Myokit](http://myokit.org/install)
2. Download the [Tor-ORd model](https://models.physiomeproject.org/e/5f1/Tomek_model_endo.cellml/view) from CellML – the file you'll need is already in this repo
3. Run `convert_to_mmt.py` to convert `Tomek_model_endo.cellml` to a `.mmt` file.

### Update the .mmt file to align with Myokit best practices

You should only complete this step if you converted the Tor-ORd file from CellML to .mmt.

Add the following code to the bottom of the `tor_ord.mmt` file:
```
[[protocol]]
#Level  Start    Length   Period   Multiplier
3.0      10       2        1000      0
```
5. Run `run_tor_ord.py` to run the model


Add the following on lines 49-60:
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


