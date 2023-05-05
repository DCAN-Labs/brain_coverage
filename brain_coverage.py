# Import libraries
import os
import pandas as pd
import re
import fsl.utils.run as fslrun
import nibabel
import sys
import glob

# Create argparser for study input, path to MNI mask, and subject/session list


def get_inputs(study_input, MNI_mask='MNI152_T1_2mm_brain_mask_dil.nii.gz'):
    '''Study input should be overarching project folder, which contains derivative and template subfolders'''
    # Setting up paths needed for pipeline 
    if study_input[-1] != '/':
        study_input = str(study_input + '/')
    study_dir = study_input # /spaces/ngdr/ref-data/abcd/nda-3165-2020-09/
    BIDS = os.path.join(study_dir, 'derivatives/abcd-hcp-pipeline/')
    #templates = os.path.join([template_dir, 'templates/'])
    #MNI_mask = os.path.join(templates + 'MNI152_T1_2mm_brain_mask.nii.gz')
    
    # Variables for naming files created by pipeline
    task_names = ["task-MID", "task-nback", "task-SST", "task-rest"]

    # Call calulation function
    run_calculation(BIDS, MNI_mask, task_names)

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
                parent_dir = os.path.dirname(task)
                task_bn = os.path.basename(task).replace('.nii.gz','')
                prefiltered_func = os.path.join(parent_dir, task_bn + '_prefiltered_func.nii.gz')

            for t in task_names:
                # Set up varibles for naming files 
                fmri_task = str("task-" + t)
                fmri_task_run = str("task-" + t + "_run-")
                sub_run_task = str(subject + ses + fmri_task)
                sub_run_basename = str(subject + ses + fmri_task_run)
                name = '_'.join([subject, session, task])
                sub_func_path = str(BIDS + subject + '/' + ses + '/func/')
        
                # Find files with run-XX in them and find the highest run number to determine # of runs 
                func_files = os.listdir(sub_func_path)
                runs = []
                for fname in func_files:
                    finding = re.findall("run-(\d+)", fname)
                    if finding != []:
                        runs.append(int(finding[0]))
                runs = sorted(runs)
                max_run = runs[-1]

                # Perform calulations for each run
                for r in max_run:
                    r = str(r)
                    volume_run = str(sub_func_path + sub_run_basename + r + "_space-MNI_bold.nii.gz")
                    if os.path.exists(volume_run):
                        # Gathering paths to files needed for coverage calculations 
                        bold_vol = str(sub_func_path + sub_run_basename + r + '_space-MNI_bold.nii.gz')
                        prefiltered_func = str(sub_func_path + sub_run_basename + r + '_space-MNI_bold_prefiltered_func.nii.gz')
                        mean_func = str(sub_func_path + sub_run_basename + r + '_space-MNI_bold_mean_func.nii.gz')    
                        mean_func_mask = str(sub_func_path + sub_run_basename + r + '_mean_func_mask.nii.gz')
                        mean_func_mask_masked = str(sub_func_path + sub_run_basename + r + '_mean_func_mask_masked.nii.gz')

                        # FSL calls to perform necessary calculations 
                        fslrun.runfsl(['fslmaths', bold_vol, prefiltered_func, '-odt', 'float'])
                        fslrun.runfsl(['fslmaths', prefiltered_func, '-Tmean', mean_func])
                        fslrun.runfsl(['fslmaths', mean_func, '-bin', mean_func_mask])
                        fslrun.runfsl(['fslmaths', mean_func_mask, '-mas', MNI_mask, mean_func_mask_masked])
                        num_vox = nibabel.imagestats.count_nonzero_voxels(mean_func_mask_masked)
                        num_temp_vox = nibabel.imagestats.count_nonzero_voxels(MNI_mask)

                        num_vox_val = float(num_vox)
                        temp_vox_val = float(num_temp_vox)
                        perc_vox_cov = (num_vox_val/temp_vox_val)*100
                        round_perc_vox = round(perc_vox_cov, 3)

                        # Add data from each run to dataframe and then output after all subjects
                        subset_data = str("derivatives.func.runs.task-" + t + "_volume")
                        data = {'participant_id': subject, "session_id": session, "data_subset": subset_data, "task": str("task-" + t), "run": str("run-" + r), "path": volume_run, 'rounded brain coverage %': round_perc_vox}
                        df = df.append(data, ignore_index=True)            
                    else:
                        print('No run ', r, ' found for', sub_run_task)

    df.to_csv(BIDS + '_brain_coverage.tsv', sep='\t', encoding='utf-8', header = None, index=False)

if __name__ == '__main__':
    get_inputs(sys.argv[1])

              

