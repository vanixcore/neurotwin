# NeuroTwin: Results Log

*Raw outputs pasted from the notebooks, unedited. Interpretation goes in the report, not here. Design and decision rules are in `research_plan.md`.*

Rules for this file: paste outputs exactly as printed; note the date and the notebook cell; never delete a result because it looks bad; if a cell is rerun, add the new output below the old one and say why.

---

## 1. Environment
**Source:** Cell 1 output of `NeuroTwin_real.ipynb`, and `import nilearn; print(nilearn.__version__)` in `01_data_prep.ipynb`.
**Date run:** 2026-11-09 (from the nvidia-smi timestamp in the notebook's setup cell)
```
PyTorch: 2.14.1+cu130 | Device: cuda
nilearn: 0.14.1
```

## 2. Data preparation (`01_data_prep.ipynb`)

*Sections 2.1-2.4 are the outputs of the original 155-subject run that produced the data used in all experiments (copied from the outputs as originally printed). Section 2.5 is a later rerun with 40 subjects that overwrote the saved files; it was not used for any result.*

### 2.1 Parcellation check (155 subjects)
**Source:** the stack-and-check cell (time points min/max, X shape, NaNs, zero-variance regions, child/adult counts).
```
time points: min 168 max 168
X shape: (155, 168, 100)
NaNs: False | zero-variance regions: 0
Child_Adult
child    122
adult     33
Name: count, dtype: int64
```

### 2.2 Head motion summary (155 subjects)
**Source:** the motion cell (overall `mean_fd` describe, by-group table, top-10 table).
```
count    155.000000
mean       0.331581
std        0.268960
min        0.063752
25%        0.164219
50%        0.242299
75%        0.387503
max        1.536827
Name: mean_fd, dtype: float64
Child_Adult                adult       child
mean_fd         count  33.000000  122.000000
                mean    0.187079    0.370668
                std     0.092742    0.287331
                min     0.063752    0.077214
                25%     0.121519    0.193854
                50%     0.167295    0.271774
                75%     0.231826    0.430533
                max     0.528821    1.536827
frac_over_0.5mm count  33.000000  122.000000
                mean    0.033730    0.162910
                std     0.068802    0.162262
                min     0.000000    0.000000
                25%     0.005952    0.059524
                50%     0.011905    0.119048
                75%     0.029762    0.190476
                max     0.375000    0.857143
               mean_fd  frac_over_0.5mm Child_Adult        Age
sub-pixar046  0.923909         0.577381       child   5.970000
sub-pixar006  0.931027         0.422619       child   3.605749
sub-pixar104  0.971565         0.392857       child   8.570000
sub-pixar075  1.097017         0.553571       child   7.830000
sub-pixar078  1.144441         0.857143       child   8.600000
sub-pixar034  1.187234         0.333333       child   5.990000
sub-pixar027  1.231459         0.315476       child   3.989049
sub-pixar040  1.238936         0.708333       child   5.290000
sub-pixar028  1.267678         0.738095       child   3.534565
sub-pixar061  1.536827         0.690476       child   5.560000
```

### 2.3 Exclusion rule counts and result (155 subjects)
**Source:** the counting cell (3 options) and the apply cell output. Rule used: mean framewise displacement > 0.5 mm.
```
mean_fd > 0.5 -> excluded: 27 | by group: {'child': 26, 'adult': 1} | remaining: 128
mean_fd > 0.7 -> excluded: 14 | by group: {'child': 14} | remaining: 141
>20% volumes > 0.5mm -> excluded: 31 | by group: {'child': 30, 'adult': 1} | remaining: 124
(128, 168, 100)
Child_Adult
child    96
adult    32
Name: count, dtype: int64
```

### 2.4 Splits (128 subjects)
**Source:** the splits cell printout.
```
train 76 {'child': 57, 'adult': 19}
val 16 {'child': 12, 'adult': 4}
cal 16 {'child': 12, 'adult': 4}
test 20 {'child': 15, 'adult': 5}
{'0.1': 8, '0.25': 19, '0.5': 38, '1.0': 76}
```

### 2.5 Later rerun with 40 subjects (NOT used for any result)
After the experiments, the data-preparation notebook was rerun with 40 subjects (most likely through a 40-subject fallback in the download cell after a failed 155-subject download), overwriting the saved `X_kept.npy` and `splits.json`. The outputs are kept below as received.


#### 2.5a Parcellation check 
**Source:** the stack-and-check cell (time points min/max, X shape, NaNs, zero-variance regions, child/adult counts).
```
time points: min 168 max 168
X shape: (40, 168, 100)
NaNs: False | zero-variance regions: 0
Child_Adult
child    31
adult     9
Name: count, dtype: int64

```

#### 2.5b Head motion summary
**Source:** the motion cell (overall `mean_fd` describe, by-group table, top-10 table).
```
count    40.000000
mean      0.328394
std       0.265120
min       0.078781
25%       0.179089
50%       0.258681
75%       0.376197
max       1.267678
Name: mean_fd, dtype: float64
Child_Adult               adult      child
mean_fd         count  9.000000  31.000000
                mean   0.214547   0.361446
                std    0.129833   0.286117
                min    0.078781   0.090253
                25%    0.148315   0.196112
                50%    0.189773   0.301785
                75%    0.234032   0.382982
                max    0.528821   1.267678
frac_over_0.5mm count  9.000000  31.000000
                mean   0.062169   0.158410
                std    0.118380   0.150252
                min    0.000000   0.000000
                25%    0.017857   0.071429
                50%    0.017857   0.125000
                75%    0.041667   0.184524
                max    0.375000   0.738095
               mean_fd  frac_over_0.5mm Child_Adult        Age
sub-pixar004  0.379028         0.190476       child   4.473648
sub-pixar001  0.386937         0.160714       child   4.774812
sub-pixar021  0.392830         0.190476       child   4.364134
sub-pixar031  0.428981         0.261905       child   4.084873
sub-pixar016  0.490337         0.291667       child   3.622177
sub-pixar012  0.514644         0.375000       child   3.786448
sub-pixar124  0.528821         0.375000       adult  33.440000
sub-pixar006  0.931027         0.422619       child   3.605749
sub-pixar027  1.231459         0.315476       child   3.989049
sub-pixar028  1.267678         0.738095       child   3.534565

```

#### 2.5c Exclusion rule counts and result
**Source:** the counting cell (3 options) and the apply cell output (kept shape and child/adult counts). Rule used: mean framewise displacement > 0.5 mm.
```
mean_fd > 0.5 -> excluded: 5 | by group: {'child': 4, 'adult': 1} | remaining: 35
mean_fd > 0.7 -> excluded: 3 | by group: {'child': 3} | remaining: 37
>20% volumes > 0.5mm -> excluded: 7 | by group: {'child': 6, 'adult': 1} | remaining: 33
(35, 168, 100)
Child_Adult
child    27
adult     8
Name: count, dtype: int64

```

#### 2.5d Splits
**Source:** the splits cell printout (train/val/cal/test counts and subset sizes).
```
train 9 {'child': 7, 'adult': 2}
val 8 {'child': 6, 'adult': 2}
cal 8 {'child': 6, 'adult': 2}
test 10 {'child': 8, 'adult': 2}
{'0.1': 1, '0.25': 2, '0.5': 4, '1.0': 9}

```

## 3. Model setup (`NeuroTwin_real.ipynb`)

### 3.1 Final configuration
**Source:** Cell 2 `print(cfg)` output with `horizon=10`, and Cell 7 `Parameters:` line.
```
Cfg(n_subjects=128, n_regions=100, n_latents=4, T=168, ctx=64, horizon=10, stride=8, d_model=64, heads=4, layers=2, dropout=0.1, batch=64, pre_epochs=15, ft_epochs=20, lr=0.001, mask_ratio=0.3, patch=4)
Parameters: 186380

```

### 3.2 Windows
**Source:** Cell 5 printout (windows per split, no-overlap check).
```
train:  76 subjects -> windows (912, 74, 100)
val  :  16 subjects -> windows (192, 74, 100)
cal  :  16 subjects -> windows (192, 74, 100)
test :  20 subjects -> windows (240, 74, 100)
No subject overlap between any splits: OK

```

## 4. Horizon decision (rule in research_plan.md section 4)

### 4.1 Ridge grid and per-step validation R2 with horizon = 16
**Source:** the ridge cell output when `cfg.horizon` was 16 (top 5 rows, `horizon in current data` line, per-step R2 list).
```
 p   alpha  val_mse
 8 10000.0 0.916239
 4 10000.0 0.924829
 1  1000.0 0.932074
 2 10000.0 0.932258
 2  1000.0 0.935260
horizon in current data: 16
ridge val R2 per step: [0.13  0.056 0.044 0.031 0.035 0.035 0.03  0.033 0.038 0.021 0.019 0.008
 0.031 0.037 0.031 0.029]
```

### 4.2 Same output after setting horizon = 10 (final ridge settings)
```
 p   alpha  val_mse
 8 10000.0 0.903566
 4 10000.0 0.911968
 1  1000.0 0.915167
 2  1000.0 0.917845
 4  1000.0 0.919548
horizon in current data: 10
ridge val R2 per step: [0.129 0.056 0.045 0.031 0.035 0.035 0.03  0.033 0.038 0.021]

```

## 5. Single-seed pilot (100% training subjects, horizon 10)
**Source:** Cell 11 results table. Note: inspected on test subjects before the experiment loop.
```
                                    MAE    RMSE      R2
NeuroTwin (pretrained)           0.7667  0.9701  0.0137
Transformer (no pretraining)     0.7676  0.9710  0.0119
Baseline: persistence            1.0432  1.3240 -0.8374
Baseline: context mean           0.8012  1.0110 -0.0713
Baseline: zero (train mean)      0.7730  0.9768 -0.0000
Baseline: group-avg time course  0.7451  0.9454  0.0632
Baseline: ridge (lag-VAR)        0.7547  0.9556  0.0429

```

### 5.1 Fine-tuning curves (Cell 9 printout; horizon 10, 100% training subjects, 20 epochs) [confirm which run]
Best validation loss: pretrained 0.9299 (epoch 17), scratch 0.9340 (epoch 12). The best-validation weights are restored at the end of `finetune`.
```
[pretrained] epoch  1/20  train=0.9934  val=0.9554
[pretrained] epoch  2/20  train=0.9535  val=0.9520
[pretrained] epoch  3/20  train=0.9498  val=0.9487
[pretrained] epoch  4/20  train=0.9463  val=0.9464
[pretrained] epoch  5/20  train=0.9435  val=0.9447
[pretrained] epoch  6/20  train=0.9411  val=0.9430
[pretrained] epoch  7/20  train=0.9386  val=0.9414
[pretrained] epoch  8/20  train=0.9355  val=0.9394
[pretrained] epoch  9/20  train=0.9316  val=0.9376
[pretrained] epoch 10/20  train=0.9266  val=0.9372
[pretrained] epoch 11/20  train=0.9226  val=0.9347
[pretrained] epoch 12/20  train=0.9165  val=0.9328
[pretrained] epoch 13/20  train=0.9105  val=0.9321
[pretrained] epoch 14/20  train=0.9060  val=0.9344
[pretrained] epoch 15/20  train=0.9008  val=0.9308
[pretrained] epoch 16/20  train=0.8946  val=0.9324
[pretrained] epoch 17/20  train=0.8888  val=0.9299
[pretrained] epoch 18/20  train=0.8834  val=0.9350
[pretrained] epoch 19/20  train=0.8759  val=0.9377
[pretrained] epoch 20/20  train=0.8696  val=0.9383
[scratch] epoch  1/20  train=0.9738  val=0.9563
[scratch] epoch  2/20  train=0.9531  val=0.9518
[scratch] epoch  3/20  train=0.9481  val=0.9480
[scratch] epoch  4/20  train=0.9440  val=0.9455
[scratch] epoch  5/20  train=0.9405  val=0.9435
[scratch] epoch  6/20  train=0.9373  val=0.9420
[scratch] epoch  7/20  train=0.9336  val=0.9404
[scratch] epoch  8/20  train=0.9295  val=0.9385
[scratch] epoch  9/20  train=0.9246  val=0.9386
[scratch] epoch 10/20  train=0.9199  val=0.9365
[scratch] epoch 11/20  train=0.9137  val=0.9363
[scratch] epoch 12/20  train=0.9083  val=0.9340
[scratch] epoch 13/20  train=0.9011  val=0.9375
[scratch] epoch 14/20  train=0.8941  val=0.9383
[scratch] epoch 15/20  train=0.8861  val=0.9370
[scratch] epoch 16/20  train=0.8793  val=0.9390
[scratch] epoch 17/20  train=0.8738  val=0.9402
[scratch] epoch 18/20  train=0.8642  val=0.9468
[scratch] epoch 19/20  train=0.8560  val=0.9514
[scratch] epoch 20/20  train=0.8494  val=0.9470
```

## 6. Data-scaling experiment

### 6.1 Run log
**Source:** the loop cell printout (20 runs and elapsed time).
```
rows: 2000 | runs: 20
frac
0.10     8
0.25    19
0.50    38
1.00    76
Name: n_train, dtype: int64

```

### 6.2 Primary comparison (bootstrap CIs) and mean MSE by training size
**Source:** the analysis cell (first line `runs per (seed, frac)`, CI table, MSE table).
```
runs per (seed, frac): 20 | rows: 2000

frac | mean diff | 95% bootstrap CI over subjects
 0.1 | +0.0020 | [+0.0017, +0.0024]
0.25 | +0.0029 | [+0.0021, +0.0037]
 0.5 | +0.0005 | [-0.0021, +0.0030]
 1.0 | +0.0031 | [+0.0012, +0.0049]

arm   scratch  pretrained   ridge  group_avg    zero
frac                                                
0.10   0.9649      0.9628  0.9470     0.9906  0.9541
0.25   0.9593      0.9564  0.9357     0.9288  0.9541
0.50   0.9519      0.9514  0.9257     0.9061  0.9541
1.00   0.9446      0.9415  0.9131     0.8939  0.9541

```

### 6.3 Per-seed consistency
**Source:** the per-seed table cell.
```
seed       0       1       2       3       4
frac                                        
0.10  0.0020  0.0033  0.0024  0.0009  0.0017
0.25  0.0024  0.0026  0.0036  0.0025  0.0034
0.50  0.0036  0.0037  0.0020 -0.0049 -0.0018
1.00  0.0048  0.0028  0.0043  0.0005  0.0030

```

### 6.4 Scaling plot
**File:** 
https://github.com/vanixcore/neurotwin/blob/main/results/result/scaling_plot_result.png
**Caption:** Data-scaling: test MSE vs number of training subjects

## 7. Uncertainty calibration

### 7.1 MC-dropout coverage
**Source:** Cell 15 output (coverage line) and the "Forecast with uncertainty" figure. Note which test windows were used (first 200 or all 240).
**Windows used:** first 200 of 240 test windows, 30 MC-dropout samples (from Cell 15 code).
```
 Fraction of true values inside +/-2 std band: 5.91%  (well-calibrated ~95%)

```

### 7.2 Conformal calibration
**Source:** the conformal cell output (zero predictor, Transformer, ridge).
```
Zero predictor (no model): coverage 94.6% | mean interval width 3.770
   per-subject coverage: min 93.3%, median 94.4%, max 95.9%
   children 94.7% | adults 94.2%
Transformer (pretrained): coverage 94.3% | mean interval width 3.711
   per-subject coverage: min 92.8%, median 94.3%, max 95.9%
   children 94.4% | adults 94.1%
Ridge: coverage 94.4% | mean interval width 3.669
   per-subject coverage: min 92.5%, median 94.4%, max 96.1%
   children 94.5% | adults 94.1%

```

## 8. Optional extras (paste only if run)

### 8.1 Lag-1 autocorrelation
```
mean lag-1 autocorrelation: 0.4138612173789571

```

### 8.2 Other figures saved
https://github.com/vanixcore/neurotwin/tree/main/results/result

## 9. Run notes
Data-prep notebook rerun with 40 subjects overwrote X_kept.npy and splits.json after the experiments; regeneration with 155 subjects not completed or re-verified. Results in runs.csv predate the overwrite
