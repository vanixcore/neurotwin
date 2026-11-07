# NeuroTwin: Research Plan

*Learning project. Not a medical tool. Inspired by the Stanford Digital Brain Project; no affiliation or connection.*

Status: plan written before running real-data experiments. Items marked **[confirm]** are to be checked once the real dataset is loaded.

## 1. Research question

When only a small number of real fMRI subjects are available, does masked-reconstruction pretraining improve forecasting of held-out subjects compared with training from scratch? And is any benefit (or harm) concentrated in particular functional brain networks?

**Why this question.** Existing fMRI foundation-model work (e.g. BrainLM, Brain-JEPA) pretrains on tens of thousands of scans. Whether masked pretraining helps at small scale is, as far as I have read, not tested in those papers. Time-series work (PatchTST) reports mixed results for pretraining on small datasets. This project tests a small piece of the "how much data is enough?" question on free compute.

## 2. Hypotheses

- **H0 (primary):** Pretraining makes no difference to held-out forecast error.
- **H1 (primary):** Pretraining reduces error, and the benefit grows as the number of training subjects shrinks (10% > 25% > 50% > 100%).
- **H2 (linked, exploratory):** If there is a pretraining effect, it differs across the 7 Schaefer functional networks.

My prior expectation is that H0 holds at most training sizes. A null result is a valid finding.

## 3. Data

- Public fMRI dataset (planned: nilearn `fetch_development_fmri`, derived from OpenNeuro ds000228; movie-watching task). **[confirm]** license, citation, number of subjects used, and number of time points per subject.
- Parcellated with the Schaefer atlas (100 regions) into an array of shape `(subjects, time, regions)`.
- Per-region z-scoring and band-pass filtering handled once, in the notebook's preprocessing step (not duplicated in the masker).
- Synthetic-data results from the original notebook are never presented as real measurements.

## 4. Design

**Splits (by subject, never by window).**
- Fixed test subjects shared across all experiments.
- Separate calibration subjects (for conformal prediction, kept apart from the early-stopping validation set).
- Remaining subjects form the training pool, from which nested subsets are drawn: 10% ⊂ 25% ⊂ 50% ⊂ 100%.
- Draft split: roughly 60% train / 15% validation / 10% calibration / 15% test. **[confirm]** against the actual subject count.

**Arms (identical except for one stage).**
- *Scratch:* train the forecasting model directly.
- *Pretrained:* masked-reconstruction pretraining, then fine-tune for forecasting.
- Same architecture, same fine-tuning budget, same hyperparameter search budget for both arms.
- Pretraining uses only the training subjects at that fraction. No extra data, and never test or calibration subjects.

**Seeds.** 5 seeds per setting. The random subset draw also varies by seed.

**Forecast setup.** Context and horizon taken from the notebook (default 64 context, 16 horizon). **[confirm]** against scan length; reduce context if the series is short.

## 5. Primary outcome and decision rule (fixed before running)

- **Metric:** test MSE of the horizon-step forecast, computed per held-out subject.
- **Comparison:** paired per-subject difference (scratch minus pretrained), averaged over seeds, with a bootstrap 95% confidence interval across subjects.
- **Verdict:** pretraining "helps" at a training size only if the CI excludes zero in favor of pretraining. Anything else is reported as no detectable benefit (or harm, if the CI excludes zero the other way).
- **Secondary metrics (descriptive):** R², error by forecast step, mask-ratio ablation.

## 6. Baselines (same splits, same metrics)

Persistence (last value), context mean, ridge or VAR, and a small LSTM. The Transformer is judged against these, not just against itself.

## 7. Linked analysis: functional networks

Uses the same trained models and test subjects as the primary analysis. No extra training.

1. Map each of the 100 parcels to its Schaefer network (7 networks) from the atlas labels. **[confirm]** label format in the installed nilearn version.
2. Average per-region error within each network, then compute the scratch-minus-pretrained difference per network.
3. Report two things per network: how predictable it is (error or R² of the pretrained model), and how much pretraining changes it.

**Rules to keep it honest.**
1. The network analysis cannot rescue a null primary result. If the overall effect is null, the network breakdown is descriptive only.
2. Run it only at the smallest and the full training fraction (2 settings × 7 networks = 14 comparisons).
3. Correct for multiple comparisons (Holm), or label results exploratory and report CIs without significance claims.
4. Stay descriptive: no claims about cognition or disease. Because the task is movie-watching, differences between networks may reflect the stimulus.

## 8. Uncertainty calibration

The notebook's MC-dropout intervals covered only about 29% of true values (target about 95%). Fix with split conformal prediction using the separate calibration subjects. Report coverage before and after, overall and per subject, because the guarantee holds only on average and windows within a subject are correlated.

## 9. Statistics and reproducibility

- Report mean ± SD or bootstrap CIs; paired comparisons across subjects.
- Fixed and reported seeds; config files saved for every run.
- Checkpoints and outputs saved to persistent storage (not only the Colab session disk).

## 10. Out of scope

- No clinical claims (no diagnosis, treatment, or condition prediction).
- No age, trait, or disease prediction.
- No numerical comparison with BrainLM or Brain-JEPA results (different data and scale).
- No claim of any link to the Stanford project; wording is "inspired by", with sources.
- No identification attempts on participants; follow the dataset license and privacy terms.

## 11. Limitations I already expect

- Small dataset and short scans; results may not generalize to larger scale.
- fMRI forecasting is a low-signal task; absolute performance will likely be modest.
- Notebook masking keeps masked steps inside the encoder (BERT/PatchTST-style), not MAE's visible-only encoder.
- Movie-watching data includes children and adults, so findings are not age-specific claims.
- Masking whole time steps across all regions tests temporal context more than cross-region structure.

## 12. Planned outputs

Dataset card, comparison table (baselines vs. Transformer), data-scaling plot (scratch vs. pretrained), network-level table, coverage before/after conformal calibration, and a 4-6 page report with an honest limitations section.

## 13. Key references (read; verify numbers in the papers before quoting)

- He et al., Masked Autoencoders Are Scalable Vision Learners (arXiv 2111.06377)
- Nie et al., PatchTST (arXiv 2211.14730)
- Ortega Caro et al., BrainLM (ICLR 2024)
- Dong et al., Brain-JEPA (arXiv 2409.19407)
- Angelopoulos and Bates, A Gentle Introduction to Conformal Prediction (arXiv 2107.07511)
- Schaefer et al., Local-Global Parcellation of the Human Cerebral Cortex
