# Import libraries
import os
import pandas as pd
import re
import sys
import glob
import argparse
import nibabel as nib
from nipype.interfaces import fsl
from nipype.interfaces.fsl.maths import MathsCommand
from nibabel import imagestats

# Create argparser for study input, path to MNI mask, and subject/session list
def _cli():
    """
    :return: Dictionary with all validated command-line arguments from the user
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-study_dir', required=True,  
        help=('Path to BIDS derivatives directory')
    )
    parser.add_argument(
        '-mni_template', required=True,
        help='Path to MNI template file'
    )
    parser.add_argument(
        '-subject_list',
        help=('List of subjects and sessions that you want to process (if you dont want to just run the whole folder)')
    )
    return vars(parser.parse_args())

def get_inputs():
    '''Study input should be overarching project folder, which contains derivative and template subfolders'''
    cli_args = _cli()
    study_input = cli_args["study_dir"]
    MNI_mask = cli_args["mni_template"]

    # Setting up paths needed for pipeline
    if study_input[-1] != '/':
        study_input = str(study_input + '/')
    study_dir = study_input # /spaces/ngdr/ref-data/abcd/nda-3165-2020-09/

    # List of possible task names
    task_names = ["task-MID", "task-nback", "task-SST", "task-rest"]

    # Call calulation function
    run_calculation(study_dir, MNI_mask, task_names)

def run_calculation(BIDS, MNI_mask, task_names):
    # Create empty dataframe that will be filled
    df_columns = ["participant_id", "session_id", "data_subset", "task", "run", "path", "rounded brain coverage %"]
    df = pd.DataFrame(columns=df_columns)

    # Loop through subjects
    for subject in os.listdir(BIDS):
        # Loop through a subjects sessions
        for session in os.listdir(os.path.join(BIDS, subject)):
            # Loop through all files with .nii.gz extension in the func folder
            tasks = glob.glob(os.path.join(BIDS, subject, session, 'func', '*.nii.gz'))
            for task in tasks:
                # Setting up paths for images that will be created
                parent_dir = os.path.dirname(task)
                task_bn = os.path.basename(task).replace('.nii.gz','')
                prefiltered_func = os.path.join(parent_dir, task_bn + '_prefiltered_func.nii.gz')
                mean_func = os.path.join(parent_dir, task_bn + '_bold_mean_func.nii.gz')
                mean_func_mask = os.path.join(parent_dir, task_bn + '_mean_func_mask.nii.gz')
                mean_func_mask_masked = os.path.join(parent_dir, task_bn + '_mean_func_mask_masked.nii.gz')

                print("starting calculations")
                # FSL calls to perform necessary calculations
                # Changes image to float type
                floating = MathsCommand(in_file= task, out_file= prefiltered_func, output_datatype= 'float', output_type= "NIFTI_GZ")
                floating.run()
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

                # Add data from each run to dataframe and then output after all subjects
                sections = task_bn.split("_")
                for s in sections:
                    if "task" in s:
                        t = s
                    elif "run" in s:
                        r = s
                subset_data = str("derivatives.func.runs." + t + "_volume")
                data = {'participant_id': subject, "session_id": session, "data_subset": subset_data, "task": t, "run": r, "path": task, 'rounded brain coverage %': round_perc_vox}
                # data = {'participant_id': subject, "session_id": session, 'rounded brain coverage %': round_perc_vox}
                df = df.append(data, ignore_index=True)            

    df.to_csv(BIDS + '_brain_coverage.tsv', sep='\t', encoding='utf-8', header = None, index=False)

if __name__ == '__main__':
    get_inputs()
    