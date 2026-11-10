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
(paste)
```

### 3.2 Windows
**Source:** Cell 5 printout (windows per split, no-overlap check).
```
(paste)
```

## 4. Horizon decision (rule in research_plan.md section 4)

### 4.1 Ridge grid and per-step validation R2 with horizon = 16
**Source:** the ridge cell output when `cfg.horizon` was 16 (top 5 rows, `horizon in current data` line, per-step R2 list).
```
(paste)
```

### 4.2 Same output after setting horizon = 10 (final ridge settings)
```
(paste)
```

## 5. Single-seed pilot (100% training subjects, horizon 10)
**Source:** Cell 11 results table. Note: inspected on test subjects before the experiment loop.
```
(paste)
```

## 6. Data-scaling experiment

### 6.1 Run log
**Source:** the loop cell printout (20 runs and elapsed time).
```
(paste)
```

### 6.2 Primary comparison (bootstrap CIs) and mean MSE by training size
**Source:** the analysis cell (first line `runs per (seed, frac)`, CI table, MSE table).
```
(paste)
```

### 6.3 Per-seed consistency
**Source:** the per-seed table cell.
```
(paste)
```

### 6.4 Scaling plot
**File:** `results/scaling_plot.png` (saved to Drive; also copy it into the repo).
**Notes:**

## 7. Uncertainty calibration

### 7.1 MC-dropout coverage
**Source:** Cell 15 output (coverage line) and the "Forecast with uncertainty" figure. Note which test windows were used (first 200 or all 240).
```
(paste)
```

### 7.2 Conformal calibration
**Source:** the conformal cell output (zero predictor, Transformer, ridge).
```
(paste)
```

## 8. Optional extras (paste only if run)

### 8.1 Lag-1 autocorrelation
```
(paste or write "not run")
```

### 8.2 Per-network analysis (H2)
```
(paste or write "not run: reason")
```

### 8.3 Other figures saved
List any saved figures (loss curves, per-region/per-step RMSE, embeddings) with file names:

## 9. Run notes
Anything unexpected during runs (crashes, reruns, Colab resets, changed cells), with dates:
