#!/bin/bash
#SBATCH --job-name="simulate 02-02"
#SBATCH --output=cpu_job.out
#SBATCH --error=cpu_job.err
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j.greenwood@deakin.edu.au

module purge
module load Anaconda3
source activate simulation_env

for f in configs_generated/*.yml # get each .yml config file in directory
do
  filename=$(basename -- "$f")
  python3 simulate.py "$f" > "simulation_out_$filename.txt" &# & to run in background, $ for variable, "" for spaced names
done
wait