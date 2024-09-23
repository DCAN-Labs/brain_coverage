#!/bin/bash -l

cd $1


file=`ls *run${SLURM_ARRAY_TASK_ID}*`

bash ${file}
