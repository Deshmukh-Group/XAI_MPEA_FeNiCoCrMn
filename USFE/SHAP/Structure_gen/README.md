# Structure regeneration (Step 6) — USFE

`Step6_structure_regen.py` rebuilds LAMMPS structure files for the 500 SHAP-sampled
USFE structures, recoloring atom types by SHAP sign for visualization and feeding
the Warren–Cowley analysis (`../WarrenC/`).

## Inputs (relative to this dir)
- `../X_test_M1_augmented.pt` — augmented test tensor (Zenodo; see top-level *Data availability*)
- `../random500_indices_sorted_y_list.txt` — sampled indices (from `Step1`)
- `../Shap_values/Random_shap_values_bulk_sample_*.csv` — per-structure SHAP (from `Step5`)
- `data_111_use_AM1_9t.dat` — reference 111-oriented, 3600-atom cell (header + coordinates)

## Output
- `./Save_data/data100_Gen_M1_<i>_<idx>.dat` — regenerated structures consumed by `../WarrenC/run_WaC_para.py`

## Run
```bash
python Step6_structure_regen.py
```
