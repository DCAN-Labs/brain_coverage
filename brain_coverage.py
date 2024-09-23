# Import libraries
import os
import pandas as pd
import glob
import argparse
import nibabel as nib
import shutil
from nipype.interfaces import fsl
from nipype.interfaces.fsl.maths import MathsCommand
from nibabel import imagestats

# Originally created as Jupyter Notebook by Anders Perrone
# Updated and converted to Python by rae McCollum
# Function: Takes in bold images of subject, applies the MNI mask, and calculates how much the mask covers the orignial image


# Create argparser for study input, path to MNI mask, and subject/session list
def _cli():
    """
    :return: Dictionary with all validated command-line arguments from the user
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--study_dir', required=True, dest='data',
        help='Path to BIDS valid subject directories, using wildcards (*) to designate a subset of the directory to analyze'
    )
    parser.add_argument(
        '--mni_template', required=True, 
        help='Path to MNI template file'
    )
    parser.add_argument(
        '--tsv', required=True,
        help='Path to the participants.tsv where the outputs of this script will be written to'
    )
    parser.add_argument(
        '--failed_file', default= os.path.join(os.getcwd(),"failed_bc_runs"),
        help='Path to a file to store the paths of the images that failed the calculation. Default path is the current directory with a filename of "failed_bc_runs".'
    )
    parser.add_argument(
        '--work_dir', default= '/tmp/brain-coverage/',
        help='Path to a directory to store the intermediate files that are created. These will be deleted as the script runs unless the --keep flag is specified. Default path of /tmp/brain-coverage/ is used'
    )
    parser.add_argument(
        '--keep', action='store_true',
        help=("If this flag is specified, the intermediate files created by this script will be kept, otherwise they will be automatically deleted")
    )
    return vars(parser.parse_args())

def get_inputs():
    '''Study input should be overarching project folder, which contains derivative and template subfolders'''
    cli_args = _cli()
    study_input = glob.glob(cli_args['data'])
    MNI_mask = cli_args["mni_template"]
    participant_tsv = cli_args["tsv"]
    temp_work = cli_args["work_dir"]
    keep_files = cli_args["keep"]
    failed_file = cli_args["failed_file"]
    
    # Setting up paths needed for pipeline 
    for sub in study_input:
        if sub[-1] != '/':
            sub = str(sub + '/')
    study_dir = study_input # /spaces/ngdr/ref-data/abcd/nda-3165-2020-09/

    # Call calulation function
    run_calculation(study_dir, MNI_mask, participant_tsv, temp_work, keep_files, failed_file)

def run_calculation(sub_list, MNI_mask, participant_tsv, temp_work, keep, failed_filepath): 
    # Load the existing participants.tsv into a DataFrame
    tsv = pd.read_csv(participant_tsv, sep='\t')

    # Loop through subjects 
    for subject in sub_list:
        # Loop through a subjects sessions 
        for session in os.listdir(subject):
            # Loop through all files with .nii.gz extension in the func folder
            tasks = glob.glob(os.path.join(subject, session, 'func', '*bold.nii.gz'))
            values = {}
            for task in tasks:
                # Setting up paths for images that will be created
                temp_dir = os.path.join(temp_work, subject.split('/')[-1])
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                task_bn = os.path.basename(task).replace('.nii.gz','')
                prefiltered_func = os.path.join(temp_dir, task_bn + '_prefiltered_func.nii.gz')
                mean_func = os.path.join(temp_dir, task_bn + '_bold_mean_func.nii.gz')
                mean_func_mask = os.path.join(temp_dir, task_bn + '_mean_func_mask.nii.gz')
                mean_func_mask_masked = os.path.join(temp_dir, task_bn + '_mean_func_mask_masked.nii.gz')
                extra_files = [prefiltered_func, mean_func, mean_func_mask, mean_func_mask_masked]
                
                print("starting calculations for image: ", task)
                # FSL calls to perform necessary calculations 
                # Changes image to float type 
                floating = MathsCommand(in_file= task, out_file= prefiltered_func, output_datatype= 'float', output_type= "NIFTI_GZ")
                floating.run()
                if not os.path.exists(prefiltered_func):
                    failed_sub = prefiltered_func.split('/')[3]
                    print(f"{prefiltered_func} is not an existing path. A RUN FROM THIS SUBJECT FAILED:", failed_sub)
                    with open(failed_filepath, 'a') as file:
                        file.write("This image failed:" + prefiltered_func)
                    continue
                # Takes mean of T dimension
                mean = fsl.MeanImage(in_file= prefiltered_func, dimension= 'T', out_file= mean_func, output_type= "NIFTI_GZ")
                mean.run()
                #
                bined = fsl.UnaryMaths(in_file= mean_func, operation='bin', out_file= mean_func_mask, output_type= "NIFTI_GZ")
                bined.run()
                # Apply MNI mask to binned image 
                mask = fsl.ApplyMask(in_file = mean_func_mask, mask_file= MNI_mask, out_file= mean_func_mask_masked, output_type= "NIFTI_GZ")
                mask.run()

                # Loads final masked image and MNI mask and counts their voxels
                masked_img = nib.load(mean_func_mask_masked)
                num_vox = imagestats.count_nonzero_voxels(masked_img)
                mask_img = nib.load(MNI_mask)
                num_mask_vox = imagestats.count_nonzero_voxels(mask_img)

                # Calculates brain coverage percent
                num_vox_val = float(num_vox)
                mask_vox_val = float(num_mask_vox)
                perc_vox_cov = (num_vox_val/mask_vox_val)*100
                round_perc_vox = round(perc_vox_cov, 3)

                # Grab the task and run information to add as participants.tsv column 
                sections = task_bn.split("_")
                for s in sections:
                    if "task" in s:
                        t = s 
                    elif "run" in s:
                        r = s 
                
                # Name column BC_task_run-# 
                task = t.split('-')[-1]
                name = 'bc_' + task + '_' + r
                values[name] = round_perc_vox
                
                # If keep flag not specified, remove intermediate files
                if not keep:
                    shutil.rmtree(temp_dir)

            sub_id = subject.split('/')[-1]
                
            # Add columns if they don't exist
            for column in values.keys():
                if column not in tsv.columns:
                    tsv[column] = None

            # Locate the row for the subject and session and update the values
            try: 
                row_mask = (tsv['participant_id'] == sub_id) & (tsv['session_id'] == session)
                tsv.loc[row_mask, values.keys()] = [values[column] for column in values.keys()]
            except:
                print("This subject/session was not found in the participants.tsv:", sub_id, session)


    tsv.to_csv(participant_tsv, sep='\t', index=False)

if __name__ == '__main__':
    get_inputs()

# This code currently can't be run on multiple subsets of data at once due to how tsv is written 
# Changes get overwritten if running in parallel so need different way of writing
# Write out new participants.tsv for each subset of subjects then combine later
    # Three separate stages of running:
    # Split participants.tsv so only has subset 
    # Parallel processing of running calculations on subset
    # Recombine participants.tsvs 