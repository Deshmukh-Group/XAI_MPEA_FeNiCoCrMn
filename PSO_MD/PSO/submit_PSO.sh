#!/bin/bash

#SBATCH --nodes=14
#SBATCH --ntasks-per-node=8
#SBATCH --partition=<partition>
#SBATCH -A <account>
#SBATCH -t 72:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=<email>

birds_per_node=14 # the number of birds being run on each node #TODO connect config to this variable


### load module (adjust to your environment) ###
module load <fftw_module>
module load <python_module>
export PATH=<path_to_lammps_binary_dir>:$PATH

python file_utils.py --delete
python file_utils.py --create


### get node information ###
echo $SLURM_JOB_NODELIST > tt
python nodes_info.py
rm tt

### creaste nodelist file for parallel ##
while read line; do for i in $(seq 1 $birds_per_node); do echo "$line"; done; done < nodes > nodelist.txt # $(seq 1 birds-per-node)

python PSO_ANN3.py
