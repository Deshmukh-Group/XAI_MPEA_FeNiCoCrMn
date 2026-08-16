# Code for "Data-Driven Design of Cantor-Type FeNiCoCrMn Alloys: Integrating Physics-Based Models, Explainable AI, and Experiment"

<a href="https://doi.org/10.5281/zenodo.20874699"><img role="button" tabindex="0" id="modal-1280726438-trigger" aria-controls="modal-1280726438" aria-expanded="false" class="doi-modal-trigger block m-0" src="https://zenodo.org/badge/DOI/10.5281/zenodo.20874699.svg" alt="DOI: 10.5281/zenodo.20874699"></a>

This repository contains the simulation code, datasets, and machine-learning
scripts used in the paper:

> F. Wang, A. G. Iwanicki, A. T. Sose, L. A. Pressley, T. M. McQueen, and
> S. A. Deshmukh, *"Data-Driven Design of Cantor-Type FeNiCoCrMn Alloys:
> Integrating Physics-Based Models, Explainable AI, and Experiment."*
> Manuscript submitted to *Digital Discovery* (2026).
>
> (Publication details and DOI will be added upon acceptance.)

---

## Overview

A particle-swarm-optimization (PSO)–guided molecular dynamics (MD) framework is
used to explore the FeNiCoCrMn composition space and generate a dataset of
elastic properties and unstable stacking-fault energies (USFEs). One-dimensional
convolutional neural networks (1D-CNN) are trained on atom-wise descriptors, and
SHAP together with Warren–Cowley short-range-order analysis is used to interpret
the structure–property relationships. Composition-only surrogate models provide
a structure-agnostic baseline for bulk modulus and USFE prediction directly from
the five elemental concentrations. A compact three-particle PSO/MD example is
included to demonstrate the complete forward conversion from saved LAMMPS
structures and property outputs to processed descriptors, deterministic tensor
splits, and optional CNN training.

## Repository structure

```
.
├── PSO_MD/                     # PSO-guided MD generation and forward preprocessing
│   ├── PSO/                    # PSO driver, config, SLURM/run harness
│   │   └── template_dir/       # MEAM potential, starting structures, elastic/GSFE inputs
│   ├── Demo_results/           # Three completed particles: raw MD outputs + processed data
│   │   ├── 0/, 1/, 2/
│   │   └── prepare_tensors_and_train_cnn.py  # MD structures → arrays → tensors → CNN
│   └── README.md               # Detailed PSO/MD and reproducibility instructions
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
│       ├── generate_two_figures.py         # Pair-energy/correlation publication figures
│       ├── pair_plot_data.csv              # Summary data for the 15 pair types
│       ├── combined_publication_heatmap_corrected.png
│       └── average_energy_vs_pair_count_correlation.png
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
├── Compositon_only_model/     # Composition-only baseline models for bulk modulus and USFE
│   ├── data_pso_Mn.dat       # Raw PSO table: five compositions plus six calculated properties
│   ├── ML_composition_only.py # Data cleaning, train/test split, fitting, evaluation, and plotting
│   ├── Models_Regressor.py    # RF, SGD, MLP, and Bayesian-ridge model definitions
│   ├── Bulk/                  # Bulk-modulus models, scaler, metrics, and parity plots
│   └── USFE/                  # USFE models, scaler, metrics, and parity plots
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md
```

> The complete production MD structure database and **all production
> descriptor-input tensors** (any `X*.pt`, including validation and augmented
> validation tensors) are not committed here. Three representative particle
> directories with saved MD structures are included under `PSO_MD/Demo_results/`
> for forward-workflow verification. See **Data availability** below.

## Requirements

- **LAMMPS** with the MEAM package for all MD simulations
- **Interatomic potential:** 2NN MEAM for Fe–Ni–Co–Cr–Mn (files:
  `PSO_MD/PSO/template_dir/CoNiCrFeMn.meam` and
  `PSO_MD/PSO/template_dir/library.meam`; reference: Choi et al.,
  *npj Computational Materials* **4**, 1 (2018))
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
(LAMMPS types 1–5 map to Mn/Cr/Co/Fe/Ni); the USFE model uses the top 25,000
structures.

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
| Most source code and analysis outputs (CSV/PNG/TXT) | — | < 1 MB each |
| Trained 1D-CNN weights — `Bulk/Train/checkpoint.pt`, `USFE/Train/checkpoint.pt` | state_dict (13 tensors) | ~6 MB each |
| Bulk targets — `Bulk/y_train.pt`, `y_test.pt`, `y_val.pt` | (18000, 5) / (5000, 5) / (2000, 5) `float32` | < 1 MB each |
| USFE targets — `USFE/y_train.pt`, `y_test.pt`, `y_val.pt` | (18000, 1) / (5000, 1) / (2000, 1) `float32` | < 1 MB each |
| Composition-only workflow — `Compositon_only_model/` | PSO input table, fitted scikit-learn models, scalers, metrics, and PDF parity plots | < 1 MB each |
| Reproducible forward example — `PSO_MD/Demo_results/` | Three PSO particles; saved 4000/3600-atom structures, MD labels, processed arrays, and conversion/training script | ~197 MB total |

> **Note:** All `X*.pt`
> files are on Zenodo (below). The committed `checkpoint.pt` weights let you run inference without them.

### On Zenodo (all `X*.pt` descriptor tensors for training and testing — DOI: <a href="https://doi.org/10.5281/zenodo.20931695"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20931695.svg" alt="DOI"></a>)


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
2. **Verify the forward MD-to-tensor workflow:** the included three-particle
   example reproduces the processed arrays directly from the saved LAMMPS data
   files, then creates deterministic 72:8:20 train/validation/test tensors:
   ```bash
   cd PSO_MD/Demo_results
   python3 prepare_tensors_and_train_cnn.py --check-processed --prepare-only
   ```
   Use `--rebuild-processed` in place of `--check-processed` to regenerate the
   four processed text files in each particle directory. Add `--property usfe`
   for the USFE path or `--train --epochs 100` for a CNN smoke test. The example
   contains 441 records and demonstrates data provenance and execution; it is
   not intended to reproduce full-dataset model performance.
3. **Augment descriptors (M1 reordering):**
   - Elastic: `cd Bulk/DAM1 && python data_augmentation.py`
   - USFE: `cd USFE/M1 && python data_augmentation.py`
4. **Train 1D-CNN models:**
   - Elastic properties: `cd Bulk/Train && python CNN_new.py`
   - USFE: `cd USFE/Train && python CNN_usf.py`
5. **Train the composition-only baselines:** run from `Compositon_only_model/` so
   the input file is found and outputs are written to its `Bulk/` and `USFE/`
   subdirectories:
   - Bulk modulus: `python ML_composition_only.py Bulk`
   - USFE: `python ML_composition_only.py USFE`
   - For a quicker fit without repeated cross-validation, append `--skip-cv`.
   The script restores the 5 at.% lower bound to the five composition columns,
   removes invalid and repeated compositions, applies a fixed 80/20 train/test
   split, and compares random forest, SGD, MLP, and Bayesian-ridge regressors.
6. **Interpretability (SHAP):** run `Step1 … Step8` in order inside `Bulk/SHAP/`
   (and `USFE/SHAP/`); per-layer analysis via `USFE/SHAP/layer/layer_analysis_overall_use.py`
7. **Pair-interaction publication figures:** `python Bulk/Analyisis/generate_two_figures.py`
8. **Reproduce figures:**
   - Combined pair-energy/correlation heatmap and energy-correlation scatter plot:
     `python Bulk/Analyisis/generate_two_figures.py`
   - Layer-wise SHAP overall analysis: `python USFE/SHAP/layer/layer_analysis_overall_use.py`
   - Warren-Cowley SRO 5x5 panel plots: `python Bulk/SHAP/WarrenC/ana2.py`
   - SHAP summary analyses: `python Bulk/SHAP/Step7_Shap_analysis.py` and `python USFE/SHAP/Step7_Shap_analysis.py`

## Citation

If you use this code or the dataset, please cite the paper:

```bibtex
@unpublished{Wang_FeNiCoCrMn,
  title  = {Data-Driven Design of Cantor-Type FeNiCoCrMn Alloys: Integrating Physics-Based Models, Explainable AI, and Experiment},
  author = {Wang, Fangxi and Iwanicki, Allana G. and Sose, Abhishek T. and Pressley, Lucas A. and McQueen, Tyrel M. and Deshmukh, Sanket A.},
  note   = {Manuscript submitted to Digital Discovery},
  year   = {2026}
}
```
<!-- On acceptance: change to @article, add journal/volume/doi. -->

You can also cite the Zenodo software repository and training dataset:

```bibtex
@software{Wang_FeNiCoCrMn_Code,
  author       = {Wang, Fangxi and Iwanicki, Allana G. and Sose, Abhishek T. and Pressley, Lucas A. and McQueen, Tyrel M. and Deshmukh, Sanket A.},
  title        = {Code for "Data-Driven Design of Cantor-Type FeNiCoCrMn Alloys: Integrating Physics-Based Models, Explainable AI, and Experiment"},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20874698},
  url          = {https://doi.org/10.5281/zenodo.20874698}
}

@dataset{Wang_FeNiCoCrMn_Data,
  author       = {Wang, Fangxi and Iwanicki, Allana G. and Sose, Abhishek T. and Pressley, Lucas A. and McQueen, Tyrel M. and Deshmukh, Sanket A.},
  title        = {Training and validation descriptor tensors for "Data-Driven Design of Cantor-Type FeNiCoCrMn Alloys: Integrating Physics-Based Models, Explainable AI, and Experiment"},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20931695},
  url          = {https://doi.org/10.5281/zenodo.20931695}
}
```



## License

This code is released under the MIT License — see [LICENSE](LICENSE).

## Acknowledgements

Supported by NSF GlycoMIP (DMR-1933525) and PARADIM (DMR-2039380); computational
resources provided by Advanced Research Computing (ARC) at Virginia Tech.

## Contact

Fangxi Wang (fxwang@vt.edu) · Sanket A. Deshmukh (sanketad@vt.edu)
