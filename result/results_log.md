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

### 2.1 Parcellation check (155 subjects)
**Source:** the stack-and-check cell (time points min/max, X shape, NaNs, zero-variance regions, child/adult counts).
```
(paste)
```

### 2.2 Head motion summary
**Source:** the motion cell (overall `mean_fd` describe, by-group table, top-10 table).
```
(paste)
```

### 2.3 Exclusion rule counts and result
**Source:** the counting cell (3 options) and the apply cell output (kept shape and child/adult counts). Rule used: mean framewise displacement > 0.5 mm.
```
(paste)
```

### 2.4 Splits
**Source:** the splits cell printout (train/val/cal/test counts and subset sizes).
```
(paste)
```

## 3. Model setup (`NeuroTwin_real.ipynb`)

### 3.1 Final configuration
**Source:** Cell 2 `print(cfg)` output with `horizon=10`, and Cell 7 `Parameters:` line.
```
Cfg(n_subjects=128, n_regions=100, n_latents=4, T=168, ctx=64, horizon=10, stride=8, d_model=64, heads=4, layers=2, dropout=0.1, batch=64, pre_epochs=15, ft_epochs=20, lr=0.001, mask_ratio=0.3, patch=4)

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


**Notes:**

## 7. Uncertainty calibration

### 7.1 MC-dropout coverage
**Source:** Cell 15 output (coverage line) and the "Forecast with uncertainty" figure. Note which test windows were used (first 200 or all 240).
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
List any saved figures (loss curves, per-region/per-step RMSE, embeddings) with file names:

## 9. Run notes
Anything unexpected during runs (crashes, reruns, Colab resets, changed cells), with dates:
