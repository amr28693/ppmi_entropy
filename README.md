# Entropy Decomposition of PPMI MDS-UPDRS Trajectories

Analysis code for a Shannon-entropy decomposition of item-level MDS-UPDRS score distributions in the Parkinson's Precision Medicine Initiative (PPMI, formerly the Parkinson's Progression Markers Initiative). For each MDS-UPDRS Part and visit, the method computes the Shannon entropy of the population-level distribution of item scores and tests whether entropy trajectories diverge between groups in ways a total-score comparison cannot detect.

The analysis accompanying the preprint compares GBA1-associated PD against
sporadic PD, stratifying the diagnosed PD cohort by GBA1 carrier status. This is
the `h1` analysis in the commands below.

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `pandas`, `matplotlib`

```bash
pip install numpy scipy pandas matplotlib
```

## Data

The PPMI data are not redistributed in this repository. Download them from the
PPMI portal (<https://www.ppmi-info.org/access-data-specimens/download-data>)
(Data Use Agreement required) and place the CSV files in a `data/`
directory alongside the scripts. Files are located by name pattern; the
following (as downloaded on 2026-06-23) are used:

| Role | File (pattern) |
|---|---|
| Participant status | `PPMI_PD_Variants_Genetic_Status_*.csv` |
| Genetic consensus | `iu_genetic_consensus_*.csv` (Consensus APOE Genotype and Pathogenic Variants) |
| Participant genetic status | `Participant_Genetic_Status_*.csv` |
| MDS-UPDRS Part I (clinician) | `MDS-UPDRS_Part_I_*.csv` |
| MDS-UPDRS Part I (patient) | `MDS-UPDRS_Part_I_Patient_Questionnaire_*.csv` |
| MDS-UPDRS Part II | `MDS_UPDRS_Part_II__Patient_Questionnaire_*.csv` |
| MDS-UPDRS Part III | `MDS-UPDRS_Part_III_*.csv` |
| MDS-UPDRS Part IV | `MDS-UPDRS_Part_IV__Motor_Complications_*.csv` |

## Reproducing the paper

The defaults match the paper's settings (5,000 permutations, minimum 15
observations per scored value per visit, seed 42). The flags below are shown
for clarity:

```bash
# Genotype comparison (GBA1-PD vs sporadic PD): 5000 permutations, min 15 obs/per scored value/visit
python ppmi_entropy_analysis.py --analysis h1 --n-perms 5000 --min-n 15
```

Each run writes, to `results/`:

- `summary_h1.txt`: human-readable domain-level results
- `divergence_permtest_h1.json`: permutation statistics (observed IAD, null mean/SD, p-value)
- `entropy_trajectories_h1.json`: per-group, per-Part entropy at each visit
- figures (entropy trajectories and per-Part divergence)

## Robustness and helper scripts

**`annual_only_sensitivity.py`** reruns the H1 genotype comparison on annual
visits only, dropping the sparse 6-month interim windows, to confirm that the
reported divergences do not depend on low-completion interim samples. It also
reruns the full data as a self-check, so the annual filter is the only variable
that differs.

```bash
python annual_only_sensitivity.py
```

**`check_visit_counts.py`** reports per-group, per-Part, per-window participant
counts, for checking the visit-completion figures.

```bash
python check_visit_counts.py --months 60 66 72
```
(months 60, 66, and 72 span the interim completion dip discussed in the preprint;
any valid month values can be passed)

**`entropy_vs_sum.py`** reruns the H1 genotype comparison with a
conventional per-Part score metric (the mean item score, a Part-score analog)
alongside entropy, on the same data and the same permutations (same seed,
participant-level shuffles, and minimum-observation threshold). This tests the
domain-specific pattern against the choice of divergence statistic. It writes
`entropy_vs_sum_h1.{txt,json,tex}` to `results/`, where the `.tex` file is a
ready-to-include table.

```bash
python entropy_vs_sum.py --n-perms 5000 --min-n 15
```

**`pd_vs_hc.py`** runs the same entropy permutation test on all Diagnosed PD
versus Healthy Controls, restricted to Parts I-III (Part IV is excluded because
healthy controls do not receive levodopa, so motor complications are not
assessed). Motor examination (Part III) is diagnostic between these two groups,
so it should diverge here, which is the reverse of the genotype comparison,
where Part III is null. Because Part III diverges here, its null in the genotype
analysis reflects the data rather than a limitation of the method. The script
defines its own PD-vs-HC group assignment and reuses the main module's loaders,
month-snapping, and permutation test. It writes `summary_pd_vs_hc.txt`,
`divergence_permtest_pd_vs_hc.json`, `entropy_trajectories_pd_vs_hc.json`, and
figures to `results/`.

```bash
python pd_vs_hc.py --n-perms 5000 --min-n 15
```

These helpers import the main module and reuse its loading and month-snapping
functions, along with its group-assignment and permutation test where
applicable, so their results stay consistent with the main analysis.

## Reproducibility

Results use a fixed random seed (`42`) and 5,000 permutations. The
minimum-observation threshold is 15 per scored value per visit. Part-level entropy is
computed when at least ⌊n/2⌋ of a Part's scored values meet the threshold. The
permutation p-value is the add-one estimator, (b + 1) / (m + 1).

## Scope

This repository covers the genotype analysis (GBA1-PD vs. sporadic PD) reported
in the preprint. Molecular stratification (for example, by alpha-synuclein seed
amplification assay status) is intended for future work.

## Data provenance and acknowledgment

Data used in the preparation of this code were obtained on 2026-06-23 from the
Parkinson's Precision Medicine Initiative (PPMI; RRID:SCR_006431),
<https://www.ppmi-info.org>. PPMI, a public-private partnership, is funded by the
Michael J. Fox Foundation for Parkinson's Research and funding partners (listed
at <https://www.ppmi-info.org>). Use of PPMI data is governed by the PPMI Data
Use Agreement, and the data themselves are not included in this repository.

## License

The analysis code is released under the MIT License. PPMI's Data Use Agreement
governs the data described above, not this code.
