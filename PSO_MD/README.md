# PSO-guided MD dataset generation and CNN preprocessing

This directory contains the particle-swarm-optimization (PSO) and LAMMPS
workflow used to explore the FeNiCoCrMn composition space, together with a
three-particle example that reproduces the forward conversion from saved MD
structures and calculated properties to CNN-ready tensors.

## Repository layout

```text
PSO_MD/
├── PSO/                              # PSO driver and cluster run harness
│   ├── PSO_ANN3.py                   # main PSO driver
│   ├── config.json                   # swarm and composition-space settings
│   ├── submit_PSO.sh                 # SLURM entry point
│   ├── PSO_bash.sh                   # launches elastic and GSFE calculations
│   ├── file_utils.py                 # creates/deletes particle directories
│   ├── replace_values.sh             # inserts the proposed composition
│   ├── save_initial_structure.sh     # archives structures by replica/epoch
│   └── template_dir/                 # copied into each PSO particle directory
│       ├── data_100.dat              # 4000-atom starting structure
│       ├── data_111.dat              # 3600-atom starting structure
│       ├── HEA.py, HEA111.py         # random-alloy structure generation
│       ├── elastic1/, elastic2/, elastic3/
│       └── GSFE1/, GSFE2/, GSFE3/
├── Demo_results/                     # three representative PSO particles
│   ├── 0/, 1/, 2/                    # raw MD results + processed text data
│   └── prepare_tensors_and_train_cnn.py
├── prepare_tensors_and_train_cnn.py  # original production-data training script
└── README.md
```

The LAMMPS atom-type mapping used throughout the workflow is:

| Atom type | Element |
|---:|:---|
| 1 | Mn |
| 2 | Cr |
| 3 | Co |
| 4 | Fe |
| 5 | Ni |

## PSO/MD workflow

The PSO proposes a five-component composition. For every particle and epoch,
the workflow creates three independently randomized atomic configurations and
runs elastic-property and generalized-stacking-fault-energy calculations:

```text
PSO composition
  -> three randomized (100) and (111) structures
  -> LAMMPS elastic1/2/3 and GSFE1/2/3 calculations
  -> saved structures and extracted property files
  -> atomic-type arrays and property labels
  -> train/validation/test tensors
  -> 1D-CNN
```

### Composition constraint

The search is constrained to physically valid five-component compositions:

- Each PSO variable is bounded by `min_var` and `max_var` in `config.json`.
- `optimize1()` projects the five variables so their sum is `total = 75`.
- The structure-generation scripts add a 5 at.% baseline for each element.
- The resulting compositions sum to 100 at.% and each element lies between
  approximately 5 and 35 at.%.

The principal configuration values are:

| Key | Current value | Meaning |
|:---|:---|:---|
| `num_birds` | 196 | swarm size |
| `dim` | 5 | number of composition variables |
| `nresults` | 6 | five elastic targets plus USFE |
| `total` | 75 | constrained sum before the 5 at.% baselines |
| `max_epochs` | 200 | maximum PSO iterations |
| `w` | 0.729 | inertia coefficient |
| `c1`, `c2` | 1.49445 | cognitive and social coefficients |

### Running the production PSO workflow

The production scripts are designed for a SLURM cluster with LAMMPS and GNU
parallel. First replace the `<placeholders>` in `submit_PSO.sh` and
`PSO_bash.sh` with the local account, partition, modules, and LAMMPS executable.
Then run:

```bash
cd PSO_MD/PSO
sbatch submit_PSO.sh
```

`file_utils.py` copies `PSO/template_dir/` into numeric particle directories
under `PSO/`. The PSO driver updates their compositions, runs LAMMPS, reads the
six extracted targets, and updates the swarm. Summary results accumulate in
`datafile.dat` and `bestresults.dat`.

## Reproducible three-particle example

`Demo_results/0`, `1`, and `2` are representative completed particle
directories. They contain both sides of the forward preprocessing workflow:

- `file_save/data_100_use<replica>_<epoch>.dat`: saved 4000-atom structures;
- `file_save111/data_111_use<replica>_<epoch>.dat`: saved 3600-atom structures;
- `elastic1.txt`, `elastic2.txt`, `elastic3.txt`: elastic-property labels for
  the three randomized replicas;
- `gsfe.txt`: the three raw USFE values for each epoch;
- `type_file.txt`, `prop.txt`: processed elastic inputs and five labels;
- `type_file_usf.txt`, `prop_usf.txt`: processed USFE inputs and labels.

The processed example selects epochs 0-48: 49 epochs x 3 replicas = 147 records
per particle and 441 records across particles 0-2. Records use replica-major
ordering: replica 1 epochs 0-48, followed by replica 2 epochs 0-48, and then
replica 3 epochs 0-48. Additional later raw files are retained as run
provenance, but they are not part of this 441-record processed example.

The five columns of `prop.txt` are bulk modulus, C11, C12, C44, and Young's
modulus, in GPa. `prop_usf.txt` contains one USFE target per structure.

### Install the Python dependencies

From the repository root:

```bash
python3 -m pip install -r requirements.txt
```

The forward preprocessing example does not require rerunning LAMMPS because the
necessary saved MD structures and results are included.

### Verify the MD-to-processed-data conversion

```bash
cd PSO_MD/Demo_results
python3 prepare_tensors_and_train_cnn.py --check-processed --prepare-only
```

This command reads every selected LAMMPS data file, extracts atom types, sorts
them by atom ID, aligns them with the replica and epoch labels, and verifies that
the regenerated content exactly matches the four checked-in processed files in
each particle directory.

To regenerate those processed files before creating the tensor splits, run:

```bash
python3 prepare_tensors_and_train_cnn.py --rebuild-processed --prepare-only
```

### Tensor split

The example uses a deterministic two-stage, structure-level random split:

1. Hold out 20% of the structures for testing (`random_state=42`).
2. Assign 10% of the remaining 80% to validation (`random_state=43`).

This gives nominal fractions of 72% training, 8% validation, and 20% testing.
For the 441-record example, the exact counts are 316 training, 36 validation,
and 89 test structures. The generated tensors and `split_metadata.json` are
written to `Demo_results/generated/elastic/` by default.

Prepare the USFE tensors instead with:

```bash
python3 prepare_tensors_and_train_cnn.py \
    --property usfe \
    --check-processed \
    --prepare-only
```

### Optional CNN smoke test

Train the elastic-property demonstration model with:

```bash
python3 prepare_tensors_and_train_cnn.py --train --epochs 100
```

For the USFE model:

```bash
python3 prepare_tensors_and_train_cnn.py \
    --property usfe \
    --train \
    --epochs 100
```

The example contains only three PSO particles and is intended to demonstrate
data provenance and code execution, not to reproduce the performance of models
trained on the complete production dataset.

## Software requirements

For the complete PSO/MD workflow:

- Python 3 with NumPy and pandas;
- LAMMPS compiled with the MEAM package;
- GNU parallel;
- a SLURM scheduler, or local adaptations of the submission scripts.

For preprocessing and CNN training:

- NumPy;
- scikit-learn;
- PyTorch;
- Matplotlib.

See the repository-level `requirements.txt` for the full Python environment.
