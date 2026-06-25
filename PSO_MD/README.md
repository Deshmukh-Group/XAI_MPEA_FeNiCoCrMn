# PSO-guided MD dataset generation

Particle-swarm optimization (PSO) coupled to LAMMPS molecular dynamics, used to
explore the FeNiCoCrMn composition space and generate the elastic-constant and
unstable-stacking-fault-energy (USFE) dataset.

## Layout

```
PSO_MD/
├── PSO/                      # PSO driver + cluster run harness
│   ├── PSO_ANN3.py           # PSO driver (launched by submit_PSO.sh)
│   ├── config.json           # Swarm + search config (see below)
│   ├── submit_PSO.sh         # SLURM job script (entry point)
│   ├── PSO_bash.sh           # Builds the per-bird LAMMPS job list, runs via GNU parallel
│   ├── file_utils.py         # Create/delete per-bird working directories
│   ├── nodes_info.py         # Parse SLURM nodelist -> nodes file
│   ├── replace_values.sh     # Inject composition into template inputs
│   ├── save_initial_structure.sh
│   └── bestresults.dat, result.txt, datafile.dat   # Example generated PSO-MD results
└── template_dir/             # Per-evaluation template copied for each bird
    ├── CoNiCrFeMn.meam, library.meam   # 2NN MEAM potential (Fe-Ni-Co-Cr-Mn)
    ├── data_100.dat, data_111.dat      # Initial structures ([100] and [111] orientations)
    ├── HEA.py, HEA111.py               # Random-alloy structure generation
    ├── Header100.txt, Header111.txt    # LAMMPS data-file headers
    ├── template.prm, value.sh          # Parameter template + result-extraction script
    ├── elastic1/, elastic2/, elastic3/ # Elastic-constant LAMMPS inputs (elastic.in + elas_*.mod)
    └── GSFE1/, GSFE2/, GSFE3/          # Generalized stacking-fault-energy inputs (GSFE.in)
```

## config.json

| Key | Value | Meaning |
|-----|-------|---------|
| `num_birds` | 196 | swarm size |
| `dim` | 5 | search dimensions (the 5 element fractions) |
| `nresults` | 6 | targets per evaluation |
| `target` | [300, 390, 246, 180, 250, 70] | target property values |
| `wt` | [1.0, 0.7, 0.7, 0.7, 0.7, 1.0] | per-target weights in the fitness |
| `min_var` / `max_var` | 0 / 30 | search-variable bounds (see composition constraint below) |
| `total` | 75 | constrained sum of the 5 search variables |
| `max_epochs` | 200 | PSO iterations |
| `w`, `c1`, `c2` | 0.729, 1.49445, 1.49445 | inertia / cognitive / social coefficients |

### Composition constraint (constrained sampling)

The search is constrained so every sampled composition is a valid 5-component
FeNiCoCrMn HEA summing to 100 at.%:

- Each search variable `var[i]` is bounded to `[0, 30]` (`min_var`/`max_var`).
- `optimize1()` (PSO_ANN3.py) projects every candidate so the five variables
  **sum to `total = 75`**, re-applied at initialization and after each PSO update.
- A **+5 at.% baseline is added per element** during structure generation
  (`HEA.py`: `n_i = var[i] + 5`), so each element lands in **[5, 35] at.%**.

Net result: 5 elements × 5 baseline + 75 = **100 at.%** (`N_total = 100` in
`HEA.py`). The constraint guarantees physically meaningful equiatomic-region
compositions rather than arbitrary fractions.

## Requirements

- **LAMMPS** with the MEAM package
- **GNU parallel** and a SLURM scheduler
- **Python 3** (standard library only: `json`, `math`, `random`, `subprocess`, …)

> The submission/run scripts contain `<placeholders>` for all environment-specific
> values — SLURM `<account>`, `<partition>`, `<email>`, module names
> (`<fftw_module>`, `<python_module>`), and the LAMMPS binary
> (`<lammps_binary>` / `<path_to_lammps_binary_dir>`). Fill these in for your own
> cluster before running. The node list is generated automatically by
> `submit_PSO.sh` from `$SLURM_JOB_NODELIST`.

## How to run

```bash
cd PSO_MD/PSO
sbatch submit_PSO.sh      # sets up nodes, then launches PSO_ANN3.py
```

`submit_PSO.sh` builds the node list, then `PSO_ANN3.py` drives the swarm: each
bird gets a copy of `template_dir/`, its composition is injected, LAMMPS runs the
`elastic*` and `GSFE*` jobs, and `value.sh` extracts the 6 targets that feed back
into the PSO fitness. Results accumulate in `datafile.dat` / `bestresults.dat`.

> All cluster-specific values are `<placeholders>` and must be edited before
> running elsewhere (see the note under **Requirements**).
