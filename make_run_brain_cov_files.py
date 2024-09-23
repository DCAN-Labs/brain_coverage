#!/usr/bin/env python3

import os
import pdb

# determine data directory, run folders, and run templates

data_bucket="data/bucket/"
derivatives_prefix='derivatives/abcd-hcp-pipeline/'
bids_prefix=''
data_dir="/tmp" # where to output data
run_folder=os.getcwd()
print("running code in:",run_folder)

file_mapper_cifti_connectivity_folder="{run_folder}/run_files.xcp_d".format(run_folder=run_folder)
file_mapper_cifti_connectivity_template="template.xcp-d"

# if processing run folders  exist delete them and recreate
if os.path.isdir(file_mapper_cifti_connectivity_folder):
      print("removing folder and remaking run files")
      os.system('rm -rf {file_mapper_cifti_connectivity_folder}'.format(file_mapper_cifti_connectivity_folder=file_mapper_cifti_connectivity_folder))
      os.system('mkdir -p {file_mapper_cifti_connectivity_folder}/logs'.format(file_mapper_cifti_connectivity_folder=file_mapper_cifti_connectivity_folder))
else:
    print("making run directory")
    os.system('mkdir -p {file_mapper_cifti_connectivity_folder}/logs'.format(file_mapper_cifti_connectivity_folder=file_mapper_cifti_connectivity_folder))

with open('/csv/path','r') as f:
    lines = f.readlines()
subject_list = [line.strip('\n') for line in lines]

run_file_num = 0
for row in subject_list:
    # generate file-mapper and cifti-connectivity template if subject ID is not in cifti-connectivity derivatives folder
    if 'sub-' in row:
        subj_id=row.split(',')[0].split('-')[1]
        ses_id=row.split(',')[1].split('-')[1]
        print("subject:", subj_id)
        print("session:", ses_id)
        file_mapper_cifti_connectivity_template_open=open("{run_folder}/{file_mapper_cifti_connectivity_template}".format(run_folder=run_folder,file_mapper_cifti_connectivity_template=file_mapper_cifti_connectivity_template),'rt')
        run_file = open(os.path.join(file_mapper_cifti_connectivity_folder,'run%s' %run_file_num), 'wt')
        mod1 = file_mapper_cifti_connectivity_template_open.read().replace('SUBJECTID',subj_id)
        mod2 = mod1.replace('BUCKET',"s3://" + data_bucket)
        mod3 = mod2.replace('RUNDIR',run_folder)
        mod4 = mod3.replace('DATADIR',data_dir)
        mod5 = mod4.replace('SESSIONID',ses_id)
        run_file.write(mod5)
        run_file.close()
        file_mapper_cifti_connectivity_template_open.close()
        run_file_num += 1

os.system('chmod +x -R {file_mapper_cifti_connectivity_folder}'.format(file_mapper_cifti_connectivity_folder=file_mapper_cifti_connectivity_folder))
