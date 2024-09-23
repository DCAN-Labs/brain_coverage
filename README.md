# Structural QC
The following are automated metrics for evaluating processed structural data quality:<br>
Registration and Acquisition Metrics from `brain_coverage.py` <br>
  * Proportion of BOLD acquisition cut off from brain mask (% from ventral(inferior)) – Fail if more than 10 percent cut off
  * Proportion of BOLD acquisition cut off from brain mask (% from dorsal(superior)) – Fail if more than 10 percent cut off
  
  Usage:
  1) Use the `template.brain_cov` to make the run_files for each subject and session
  2) Submit into slurm parallely using the `resources_continuous_slurm_submitter_brain_cov.sh`
  3) Finally run `merge_participants.py` to merge all the temp_tsv_files
     
*Note:* Any column that is nan/empty signifies that particular file did not exist.


    
