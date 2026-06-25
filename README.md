# Code for "Data-Driven Design of Cantor-Type FeNiCoCrMn Alloys: Integrating Physics-Based Models, Explainable AI, and Experiment"

[![DOI](https://zenodo.org/badge/1280726438.svg)](https://doi.org/10.5281/zenodo.20874698)

This repository contains the simulation code, datasets, and machine-learning
scripts used in the paper:

> F. Wang, A. G. Iwanicki, A. T. Sose, L. A. Pressley, T. M. McQueen, and
> S. A. Deshmukh, *"Data-Driven Design of Cantor-Type FeNiCoCrMn Alloys:
> Integrating Physics-Based Models, Explainable AI, and Experiment."*
> Manuscript in preparation (2026).
>
> (Journal, year, and DOI will be added upon acceptance.)

---

## Overview

A particle-swarm-optimization (PSO)–guided molecular dynamics (MD) framework is
used to explore the FeNiCoCrMn composition space and generate a dataset of
elastic properties and unstable stacking-fault energies (USFEs). One-dimensional
convolutional neural networks (1D-CNN) are trained on atom-wise descriptors, and
SHAP together with Warren–Cowley short-range-order analysis is used to interpret
the structure–property relationships.

## Repository structure

```
.
├── PSO_MD/                     # PSO-guided MD dataset generation (LAMMPS) — see PSO_MD/README.md
│   ├── PSO/                    # PSO driver (PSO_ANN3.py), config.json, SLURM/run harness
│   └── template_dir/           # Per-evaluation template: 2NN MEAM potential, structures, elastic/GSFE LAMMPS inputs
├── Bulk/                       # Elastic-property workflow (1D-CNN predicts 5 targets: C11, C12, C44, bulk & Young's modulus)
│   ├── y_train.pt, y_val.pt, y_test.pt  # Targets — (18000,5)/(2000,5)/(5000,5) float32, committed (X_train/X_val/X_test.pt → Zenodo)
│   ├── DAM1/
│   │   ├── data_augmentation.py    # Descriptor-channel reordering augmentation (M1 scheme)
│   │   └── MPEA_data_aug_1.csv     # Reordering map used by the augmentation
│   ├── Train/
│   │   ├── CNN_new.py              # 1D-CNN definition + training loop (PyTorch, early stopping)
│   │   └── checkpoint.pt           # Trained model weights (committed); X_*_M1_augmented.pt → Zenodo
│   ├── SHAP/
│   │   ├── Step1_Get_Samples.py            # Sample 500 random structures for SHAP
│   │   ├── Step2_get_y_500.py              # Collect ground-truth targets for the sample
│   │   ├── Step2_2_get_y_500_predict.py    # Collect model predictions for the sample
│   │   ├── Step3_get_sample_shap_strutcures.py  # Per-structure SHAP aggregation
│   │   ├── Step4_compare.py                # Merge indices / y-values / predictions
│   │   ├── Step5_get_sample_shap_NEW.py    # Per-element positive/negative SHAP sums
│   │   ├── Step7_Shap_analysis.py          # SHAP summary analysis
│   │   ├── Step8_Shap_analysis_per_atom.py # Per-atom SHAP analysis
│   │   ├── Structure_gen/                  # Regenerate atomic structures from sampled descriptors
│   │   │   ├── structure_regen.py          #   descriptor → LAMMPS .dat structure regeneration
│   │   │   ├── data100_AM1_Mn.dat          #   example regenerated structure
│   │   │   └── Save_data/                  #   regenerated-structure outputs
│   │   └── *.csv, *.txt                    # Sample indices and per-structure SHAP outputs
│   └── Analyisis/
│       ├── Analysis.py                     # Pair-interaction / SRO heatmap analysis
│       ├── combined_publication_heatmap.png
│       └── *.csv                           # Matched pair summaries
├── USFE/                       # Unstable stacking-fault energy workflow (1D-CNN predicts 1 target)
│   ├── y_train.pt, y_val.pt, y_test.pt     # Targets — (18000,1)/(2000,1)/(5000,1) float32, committed (X_train/X_val/X_test.pt → Zenodo)
│   ├── M1/data_augmentation.py             # Descriptor reordering augmentation (M1 scheme)
│   ├── Train/
│   │   ├── CNN_usf.py                       # 1D-CNN definition + training loop (PyTorch)
│   │   └── checkpoint.pt                     # Trained model weights
│   └── SHAP/
│       ├── Step1_Get_Samples.py … Step8_Shap_analysis_per_atom.py  # Same SHAP pipeline as Bulk
│       ├── *.csv, *.txt
│       ├── Structure_gen/                  # Regenerate atomic structures from sampled descriptors
│       │   ├── Step6_structure_regen.py    #   descriptor → LAMMPS .dat structure regeneration
│       │   ├── data_111_use_AM1_9t.dat     #   example regenerated structure
│       │   └── Save_data/                  #   regenerated-structure outputs
│       └── layer/
│           ├── layer_analysis_overall_use.py   # Per-layer SHAP analysis
│           ├── layer_analysis_MS1_Overall.png
│           └── MPEA_data_aug_1.csv
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md
```

> The raw MD trajectory database (atomic structures) and **all descriptor-input tensors** (any
> `X*.pt`, including the validation and augmented-validation tensors) are not committed here;
> see **Data availability** below.

## Requirements

- **LAMMPS** (version <e.g. 2 Aug 2023>) for all MD simulations
- **Interatomic potential:** 2NN MEAM for Fe–Ni–Co–Cr–Mn (files: `PSO_MD/template_dir/CoNiCrFeMn.meam`
  + `PSO_MD/template_dir/library.meam`; reference: Choi et al., npj Comput. Mater. 4, 1 (2018))
- **Python** >= 3.10 with (see `requirements.txt`):
  - numpy, scipy, pandas, scikit-learn
  - **pytorch** (`torch`, `torchmetrics`) — the 1D-CNN models
  - shap, joblib  (SHAP value generation / loading)
  - matplotlib, seaborn, tqdm
  - **ovito** + **WarrenCowleyParameters** — Warren–Cowley SRO analysis
  - `pytorchtools` (`EarlyStopping`) is vendored in `Bulk/Train/` and `USFE/Train/`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Data availability

The PSO-guided MD framework generated **9,862 unique valid MPEA compositions** and
**29,586 atomic structures** (elastic-property cells: 4000 atoms; USFE/GSFE cells:
3600 atoms). The 1D-CNN models were trained on atom-wise element-ID descriptors
(integer labels 1–5 for Fe/Ni/Co/Cr/Mn); the USFE model uses the top 25,000 structures.

The descriptor inputs (`X*.pt`) are `int64` tensors of shape `(n_structures, n_atoms)` —
one atom-wise element-ID row (labels 1–5) per structure. The targets (`y*.pt`) are
`float32` tensors of shape `(n_structures, n_targets)`: 5 elastic targets (C11, C12, C44,
bulk & Young's modulus) for **Bulk**, 1 USFE target for **USFE**. Splits are 18000 train /
5000 test / 2000 val.

Because GitHub limits individual files to 100 MB, the dataset is split between this
repository and a Zenodo archive:

### In this repository (GitHub)

| Item | Shape (dtype) | Approx. size |
|------|------|------|
| All code, scripts, and analysis outputs (CSV/PNG/TXT) | — | < 1 MB each |
| Trained 1D-CNN weights — `Bulk/Train/checkpoint.pt`, `USFE/Train/checkpoint.pt` | state_dict (13 tensors) | ~6 MB each |
| Bulk targets — `Bulk/y_train.pt`, `y_test.pt`, `y_val.pt` | (18000, 5) / (5000, 5) / (2000, 5) `float32` | < 1 MB each |
| USFE targets — `USFE/y_train.pt`, `y_test.pt`, `y_val.pt` | (18000, 1) / (5000, 1) / (2000, 1) `float32` | < 1 MB each |

> **Note:** no descriptor-input tensor (any `X*.pt`) is committed — they are excluded by
> `.gitignore` (`[Xx]*.pt`, which matches `X_val.pt`, `X_train.pt`, `X_test.pt`, `X_data.pt`
> and `X_val_M1_augmented.pt` at any depth). This includes the validation and augmented-validation
> tensors (`Bulk/X_val.pt`, `USFE/X_val.pt`, `Bulk/Train/X_val_M1_augmented.pt`). All `X*.pt`
> files are on Zenodo (below). The committed `checkpoint.pt` weights let you run inference without them.

### On Zenodo (all `X*.pt` descriptor tensors — DOI: `<add Zenodo DOI>`)

| Item | Shape (dtype) | Approx. size |
|------|------|------|
| Elastic validation input — `Bulk/X_val.pt` | (2000, 4000) `int64` | ~62 MB |
| Elastic augmented validation input — `Bulk/Train/X_val_M1_augmented.pt` | (2000, 4000) `int64` | ~62 MB |
| Elastic training/test inputs — `Bulk/X_train.pt`, `X_test.pt` | (18000, 4000) / (5000, 4000) `int64` | ~549 MB / ~153 MB |
| USFE validation input — `USFE/X_val.pt` | (2000, 3600) `int64` | ~55 MB |
| USFE training/test inputs — `USFE/X_train.pt`, `X_test.pt` | (18000, 3600) / (5000, 3600) `int64` | ~494 MB / ~137 MB |
| USFE full descriptor set + targets — `USFE/X_data.pt`, `USFE/y_data.pt` | (25000, 3600) `int64` / (25000, 1) `float32` | ~687 MB |

To run the validation/SHAP pipeline, download the `X_val.pt` (and `X_val_M1_augmented.pt`) tensors
from Zenodo and place them next to the committed `y_val.pt` in `Bulk/`, `Bulk/Train/`, and `USFE/`.
To retrain end-to-end, also download `X_train.pt` / `X_test.pt` and place them next to the committed
`y_train.pt` / `y_test.pt` in `Bulk/` and `USFE/`.

## How to reproduce

0. **Get the descriptor tensors:** no `X*.pt` is committed to GitHub — download them from Zenodo (see *Data availability*). For inference/SHAP, fetch `X_val.pt` (and `Bulk/Train/X_val_M1_augmented.pt`) into `Bulk/`, `Bulk/Train/`, and `USFE/`. For retraining, also fetch `X_train.pt` / `X_test.pt` into `Bulk/` and `USFE/`. (The `y_*.pt` targets and `checkpoint.pt` weights are already in the repo.)
1. **Generate the dataset (PSO-guided MD):** `cd PSO_MD/PSO && sbatch submit_PSO.sh`  (see [PSO_MD/README.md](PSO_MD/README.md); edit cluster paths/modules first)
2. **Augment descriptors (M1 reordering):**
   - Elastic: `cd Bulk/DAM1 && python data_augmentation.py`
   - USFE: `cd USFE/M1 && python data_augmentation.py`
3. **Train 1D-CNN models:**
   - Elastic properties: `cd Bulk/Train && python CNN_new.py`
   - USFE: `cd USFE/Train && python CNN_usf.py`
4. **Interpretability (SHAP):** run `Step1 … Step8` in order inside `Bulk/SHAP/`
   (and `USFE/SHAP/`); per-layer analysis via `USFE/SHAP/layer/layer_analysis_overall_use.py`
5. **Pair-interaction / Warren–Cowley SRO analysis:** `cd Bulk/Analyisis && python Analysis.py`
6. **Reproduce figures:**
   - Combined publication heatmap: `python Bulk/Analyisis/Analysis.py`
   - Layer-wise SHAP overall analysis: `python USFE/SHAP/layer/layer_analysis_overall_use.py`
   - Warren-Cowley SRO 5x5 panel plots: `python Bulk/SHAP/WarrenC/ana2.py`
   - SHAP summary analyses: `python Bulk/SHAP/Step7_Shap_analysis.py` and `python USFE/SHAP/Step7_Shap_analysis.py`

## Citation

```bibtex
@unpublished{Wang_FeNiCoCrMn,
  title  = {Data-Driven Design of Cantor-Type FeNiCoCrMn Alloys: Integrating Physics-Based Models, Explainable AI, and Experiment},
  author = {Wang, Fangxi and Iwanicki, Allana G. and Sose, Abhishek T. and Pressley, Lucas A. and McQueen, Tyrel M. and Deshmukh, Sanket A.},
  note   = {Manuscript in preparation},
  year   = {2026}
}
```
<!-- On acceptance: change to @article, add journal/volume/doi. -->



## License

This code is released under the MIT License — see [LICENSE](LICENSE).

## Acknowledgements

Supported by NSF GlycoMIP (DMR-1933525) and PARADIM (DMR-2039380); computational
resources provided by Advanced Research Computing (ARC) at Virginia Tech.

## Contact

Fangxi Wang (fxwang@vt.edu) · Sanket A. Deshmukh (sanketad@vt.edu)
