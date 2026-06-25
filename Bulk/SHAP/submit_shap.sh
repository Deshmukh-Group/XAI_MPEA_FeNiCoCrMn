#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:1
#SBATCH -A <account>
#SBATCH -t 4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=<email>

### load environment (adjust to your system) ###
module load <python_module>
source activate <conda_env>

cd $SLURM_SUBMIT_DIR

# Computes SHAP (GradientExplainer) values for the elastic 1D-CNN.
# Requires checkpoint.pt + X_train/test_M1_augmented.pt + y_train/test.pt in this dir.
# Writes shap_values.pkl (large) and per-output shap_values_*.npy.
python3 shap_10K.py
