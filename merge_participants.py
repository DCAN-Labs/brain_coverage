import pandas as pd
import os
import glob

# Setting the required directories
temp_dir = '/home/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/tmp_tsv_files_6SEP'
merged_tsv = '/home/midb-ig/shared/projects/ABCD/Automated_QC/brain_coverage/participants.tsv'

# Get a list of all TSV files in the temp directory
tsv_files = glob.glob(os.path.join(temp_dir, '*.tsv'))

# Read the existing merged TSV file
if os.path.exists(merged_tsv):
    merged_df = pd.read_csv(merged_tsv, sep='\t')
else:
    # If the merged_tsv file doesn't exist, create an empty DataFrame with the desired columns
    columns = [
        'participant_id', 'session_id', 'collection_3165', 'site', 'scanner_manufacturer',
        'scanner_model', 'scanner_software', 'matched_group', 'sex', 'White',
        'Black/African American', 'American Indian, Native American', 'Alaska Native',
        'Native Hawaiian', 'Guamanian', 'Samoan', 'Other Pacific Islander', 'Asian Indian',
        'Chinese', 'Filipino', 'Japanese', 'Korean', 'Vietnamese', 'Other Asian',
        'Other Race', 'Refuse to Answer', "Don't Know",
        'Do you consider the child Hispanic/Latino/Latina?', 'age', 'handedness',
        'siblings_twins', 'income', 'participant_education', 'parental_education',
        'anesthesia_exposure', 'pc1', 'pc2', 'pc3'
    ]
    merged_df = pd.DataFrame(columns=columns)

# Iterate through each new TSV file
for file in tsv_files:
    df = pd.read_csv(file, sep='\t')
    
    if not df.empty:
        # Update or add rows in merged_df based on participant_id and session_id
        for index, row in df.iterrows():
            pid = row['participant_id']
            sid = row['session_id']
            
            if ((merged_df['participant_id'] == pid) & (merged_df['session_id'] == sid)).any():
                for col in df.columns:
                    if col not in ['participant_id', 'session_id']:
                        merged_df.loc[(merged_df['participant_id'] == pid) & (merged_df['session_id'] == sid), col] = row[col]
            else:
                print("This subject/session was not found in the participants.tsv:", pid, sid)

merged_df.to_csv(merged_tsv, sep='\t', index=False)

print(f'Merged TSV file saved to {merged_tsv}')
