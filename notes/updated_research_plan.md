# NeuroTwin: Research Plan (updated with final design and results)

*Learning project. Not a medical tool. Inspired by the Stanford Digital Brain Project; no affiliation or connection.*

Status: original plan written before any real-data experiment (Nov 7, 2026). Amended Nov 9-10, 2026 after a single-seed pilot; every amendment is listed in section 12. 
## 1. Research question

When only a small number of real fMRI subjects are available, does masked-reconstruction pretraining improve forecasting of held-out subjects compared with training from scratch? And is any benefit (or harm) concentrated in particular functional brain networks?

**Why this question.** Existing fMRI foundation-model work (e.g. BrainLM, Brain-JEPA) pretrains on tens of thousands of scans. Whether masked pretraining helps at small scale is, as far as I read, not tested in those papers. Time-series work (PatchTST) reports mixed results for pretraining on small datasets. This project tests a small piece of the "how much data is enough?" question on free compute.

## 2. Hypotheses

- **H0 (primary):** Pretraining makes no difference to held-out forecast error.
- **H1 (primary):** Pretraining reduces error, and the benefit grows as the number of training subjects shrinks (10% > 25% > 50% > 100%).
- **H2 (linked, exploratory):** If there is a pretraining effect, it differs across the 7 Schaefer functional networks.

Prior expectation: H0 holds at most training sizes. A null result is a valid finding.

## 3. Data

- **Dataset:** OpenNeuro ds000228, movie-watching task (silent animated short film, about 5.6 min), 155 subjects (122 children aged 3-12, 33 adults), accessed through `nilearn.datasets.fetch_development_fmri` (nilearn 0.14.1), fMRIPrep-preprocessed derivatives with confounds regressed during parcellation.
- **License and citation:** **[confirm: copy exactly from the OpenNeuro dataset page; cite Richardson et al., 2018 as given there]**.
- **Parcellation:** Schaefer atlas, 100 regions, 7 networks (`NiftiLabelsMasker`), no filtering or standardisation in the masker. Array shape (subjects, time, regions) = (155, 168, 100); all subjects have 168 time points (TR 2 s).
- **Exclusion rule (pre-specified from the motion distribution before any model was trained):** subjects with mean framewise displacement > 0.5 mm are excluded: 27 of 155 (26 children, 1 adult). Retained: **128 subjects (96 children, 32 adults)**. The rule removes more children than adults, so the retained children are calmer than the full child sample.
- **Preprocessing in the model notebook:** per-subject linear detrend and per-region z-scoring. No band-pass filter (a zero-phase filter would use future samples for past values).
- Synthetic-data results from the original notebook are never presented as real measurements.

## 4. Design

**Splits (by subject, stratified by child/adult, fixed once with seed 0, never regenerated):**

| Part | Subjects | Children | Adults |
|---|---|---|---|
| Train pool | 76 | 57 | 19 |
| Validation | 16 | 12 | 4 |
| Calibration | 16 | 12 | 4 |
| Test | 20 | 15 | 5 |

Nested training subsets per seed (10% inside 25% inside 50% inside 100%): **8, 19, 38, 76 subjects**.

**Windowing:** context 64 steps, horizon **10** steps (20 s), stride 8; 12 windows per subject (912 train, 192 validation, 192 calibration, 240 test windows).

**Arms (identical except for one stage).**
- *Scratch:* train the forecasting model directly.
- *Pretrained:* masked-reconstruction pretraining (mask ratio 0.3, patch 4 time steps, loss on masked steps only), then the same fine-tuning.
- Same architecture (Transformer encoder: d_model 64, 4 heads, 2 layers, dropout 0.1), same initial weights for both arms in a given run, same fine-tuning budget. The pretrained arm additionally receives the pretraining stage, which is the thing being tested.
- Pretraining and fine-tuning use only the training subjects of that run. Test and calibration subjects are never used for training or model selection.
- Settings: batch 64, learning rate 1e-3 (pretraining) and 5e-4 (fine-tuning), AdamW, best checkpoint chosen on the fixed validation subjects.
- **Training budget** is fixed in optimizer steps, not epochs: about 300 fine-tuning steps and 225 pretraining steps, with at least the notebook's default epochs (15 pretraining, 20 fine-tuning), so small training sets are not under-trained.

**Seeds.** 5 seeds (0-4); the seed varies both model initialisation and which training subjects are drawn.

**Horizon rule (set before viewing the ridge validation results):** the horizon is the largest k such that the ridge baseline's validation R2 is at least 0.02 at every step 1..k (minimum 4). Outcome: validation R2 of ridge was at least 0.02 for steps 1-10 and 0.019 at step 11, so **horizon = 10**.

## 5. Primary outcome and decision rule (fixed before the experiment loop)

- **Metric:** test MSE of the 10-step forecast, computed per held-out subject (z-scored units).
- **Comparison:** paired per-subject difference (scratch minus pretrained), averaged over seeds, with a bootstrap 95% CI across the 20 test subjects (5,000 resamples).
- **Verdict:** pretraining "helps" at a training size only if the CI lies entirely above zero. Anything else is reported as no detectable benefit.
- **Secondary (descriptive):** R2, MAE, per-seed consistency of the sign of the difference.

## 6. Baselines (same splits, same metrics)

- Persistence (last value); context mean; zero predictor (training mean, since data are z-scored).
- **Ridge (regularised lag-VAR, direct multi-step):** features are the last p time steps; p and alpha chosen on validation MSE (grid p in {1, 2, 4, 8} , alpha in {10, 100, 1000, 1e4, 1e5}); selected p = 8, alpha = 1e4. Refit on each run's training subset.
- **Group-average time course:** predicts the future with the training subjects' mean signal at the same absolute time points. It is the bar for "learned more than the shared movie response". Refit on each run's training subset.
- An LSTM baseline was dropped for time.

## 7. Results

Results are recorded in `results_log.md` (raw outputs pasted from the notebooks) and will be summarised in the report. This plan is kept free of results so it can be read as a design document. Hypotheses are judged only by the decision rule in section 5.

## 8. Statistics and reproducibility

- Subject-level bootstrap CIs, fixed seeds (0-4), fixed split file (`splits.json`), all per-subject results saved in `results/runs.csv`.
- PyTorch 2.14.1 (CUDA 13.0), nilearn 0.14.1, Colab T4 GPU. 
- The CI reflects test-subject sampling only; it does not capture variation from training-subject draw or training randomness (see the per-seed table in `results_log.md`).

## 9. Out of scope

- No clinical claims (no diagnosis, treatment, or condition prediction).
- No age, trait, or disease prediction; no age or developmental claims from the child/adult labels.
- No numerical comparison with BrainLM or Brain-JEPA results (different data and scale).
- No claim of any link to the Stanford project; wording is "inspired by", with sources.
- No attempt to identify participants; dataset license and privacy terms followed.

## 10. Limitations

- Small public dataset (128 subjects, 168 time points); one passive-viewing task; results may not generalise.
- fMRI forecasting is a low-signal task; Transformer R2 is about 0.01.
- All subjects watch the same movie, so signals are time-locked across subjects; the group-average baseline exposes this and outperforms the Transformers from 19 training subjects.
- Motion exclusion removed more children than adults; retained children are calmer than the full sample.
- Test set has only 5 adults; no group-specific claims.
- Per-subject normalisation uses the full scan (mild look-ahead); applied equally to all models and baselines, but absolute scores may be slightly optimistic.
- A single-seed pilot was inspected on the test subjects before the design was finalised; final choices (horizon, ridge settings) were made on validation data only.
- The pretrained arm receives extra encoder optimisation, so any gain may reflect extra training or initialisation, not the masked objective itself.
- Notebook masking hides whole time steps across all regions (BERT/PatchTST-style, masked tokens kept inside the encoder), not MAE's visible-only encoder; it tests temporal context more than cross-region structure.
- Conformal calibration rests on 16 subjects, and windows and regions within a subject are correlated, so the coverage guarantee is approximate.
- Five seeds and 20 test subjects limit statistical power.

## 11. Planned outputs

Dataset card, `results_log.md`, scaling plot, calibration table, 3-4 page report, repo with notebooks (`01_data_prep.ipynb`, `NeuroTwin_real.ipynb`, original synthetic notebook), `results/runs.csv`.

## 12. Amendment log (changes after the original plan)

| Date | Change | Reason | Decided using |
|---|---|---|---|
| Nov 9 | Removed band-pass filter from preprocessing | Zero-phase filter leaks future information; TR is 2 s, not 1 s | Method reasoning, no results |
| Nov 9 | Added zero predictor and group-average baselines | Shared stimulus; weak baselines misleading on z-scored data | Method reasoning, after pilot |
| Nov 10 | Horizon changed from 16 to 10 | Rule in section 4 applied to ridge validation R2 | Validation data only |
| Nov 10 | Ridge (lag-VAR) added; LSTM dropped | Strong linear baseline; time | Validation for p, alpha |
| Nov 10 | Training budget fixed in optimizer steps; 5 seeds | Small subsets should not be under-trained | Method reasoning |

## 13. Key references 
- He et al., Masked Autoencoders Are Scalable Vision Learners (arXiv 2111.06377)
- Nie et al., PatchTST (arXiv 2211.14730)
- Ortega Caro et al., BrainLM (ICLR 2024)
- Dong et al., Brain-JEPA (arXiv 2409.19407)
- Angelopoulos and Bates, A Gentle Introduction to Conformal Prediction (arXiv 2107.07511)
- Schaefer et al., Local-Global Parcellation of the Human Cerebral Cortex
- Dataset: Richardson et al., 2018 (OpenNeuro ds000228) 
