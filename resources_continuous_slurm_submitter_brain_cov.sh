#!/bin/bash -l

#SBATCH -J bc_cont_slurm_submitter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=10gb
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pandh015@umn.edu
#SBATCH -p msismall,agsmall,ag2tb,abcc
#SBATCH -o /home/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/output_logs/bc_cont_slurm_submitter_%A.out
#SBATCH -e /home/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/output_logs/bc_cont_slurm_submitter_%A.err
#SBATCH -A midb-ig


module load python

python3 continuous_slurm_submitter.py --partition msismall agsmall ag2tb abcc --job-name brain_coverage --log-dir /home/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/output_logs/brain_coverage_slurm_submitter_logs_20SEP --run_folder run_files.brain_cov --n_cpus 8 --time_limit 10:00:00 --total_memory 30 --tmp_storage 80 --array-size 1000 --submission-interval 90 --account_name midb-ig midb_abcd --emailed_user pandh015@umn.edu
