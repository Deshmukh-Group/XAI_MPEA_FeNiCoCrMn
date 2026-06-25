# Warren–Cowley SRO analysis — USFE, Fig. 10

Computes Warren–Cowley short-range-order (WC) parameters for the SHAP-sampled
structures and correlates them with per-structure SHAP values.

## Files
- `USE_Warren.py <structure.dat>` — compute WC parameters for one structure → `WC_parameters.csv`
- `run_WaC_para.py` — batch-run `USE_Warren.py` over the 500 sampled structures, concatenate → `WC_new_full_parameters_structures.csv`
- `ana2.py` — merge WC parameters with `merged_indices_y_values_per_structure_shap.csv` and plot the 5×5 panel (**Fig. 10**)
- `WC_new_full_parameters_structures.csv` — committed WC result (lets `ana2.py` reproduce the figure directly)
- `merged_indices_y_values_per_structure_shap.csv` — per-structure SHAP values (input to `ana2.py`)
- `5x5_panel_plot.png` — reference output figure

## Reproduce
- **From the committed result:** `python ana2.py` (uses `WC_new_full_parameters_structures.csv`).
- **From scratch:** `cd ../Structure_gen && python Step6_structure_regen.py` to regenerate the sampled structures into `../Structure_gen/Save_data/`, then back here run `python run_WaC_para.py`, then `python ana2.py`.

> `run_WaC_para.py` reads `../random500_indices_sorted_y_list.txt` and the
> regenerated `../Structure_gen/Save_data/*.dat` structures.
