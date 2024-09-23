#!/bin/bash -l
# SBATCH --job-name=merge_par
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=10gb
#SBATCH -t 14:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pandh015@umn.edu
#SBATCH -p msismall,agsmall,ag2tb,abcc
#SBATCH -o /home/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/output_logs/merge_par_%A.out
#SBATCH -e /home/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/output_logs/merge_par_%A.err
#SBATCH -A midb_abcd

module load python

source /home/midb-ig/shared/projects/ABCD/fully_processed_abcd-hcp-pipeline_example/code/pythonenv/bin/activate

python3 /home/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/merge_participants.py