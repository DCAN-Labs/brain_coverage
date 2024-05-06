# Import standard libraries
import argparse
from datetime import datetime
from glob import glob
import os
import pandas as pd
import shutil
import sys

# Import installed libraries
import nibabel as nib
from nibabel import imagestats
from nipype.interfaces import fsl
from nipype.interfaces.fsl.maths import MathsCommand

# Originally created as Jupyter Notebook by Anders Perrone
# Converted to Python by rae McCollum
# Updated by Greg Conan 2024-01-30
# Function: Takes in bold images of subject, applies the MNI mask, and calculates how much the mask covers the orignial image


# Create argparser for study input, path to MNI mask, and subject/session list
def _cli():
    """
    :return: Dictionary with all validated command-line arguments from the user
    """
    DEFAULT_FAIL_FILE = os.path.join(os.getcwd(), "failed_bc_runs.txt")
    DEFAULT_WORK_DIR = "/tmp/brain-coverage/"
    MSG_DEFAULT = " By default, this argument's value will be '{}'."
    MSG_PATH = "Valid path to existing"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fail", "--failed-file", default=DEFAULT_FAIL_FILE,
        help=("Valid path to a file to store the paths of the images that "
              "failed the brain coverage calculation."
              + MSG_DEFAULT.format(DEFAULT_FAIL_FILE))
    )
    parser.add_argument(
        '--in-tsv', required=True, type=valid_readable_file,
        help=(f"{MSG_PATH} participants.tsv file that the inputs of this "
              "script will be read from.")
    )
    parser.add_argument(
        '--keep', action='store_true',
        help=("If this flag is specified, the intermediate files created by "
              "this script will be kept. Otherwise, they will be "
              "automatically deleted.")
    )
    parser.add_argument(
        '--mni-template', required=True, help=f"{MSG_PATH} MNI template file."
    )
    parser.add_argument(
        '--out-tsv', required=True,
        help='Path to the participants.tsv where the outputs of this script will be written.'
    )
    parser.add_argument(
        "--bids-dir", type=valid_readable_dir,
        help=("Valid path to existing BIDS root directory with 'sub-*' subdirectories.")
    )
    # Add participant/subject ID and session name/ID arguments
    parser = add_subj_ses_arg_to(parser, "participant", "sub-NDARINV")
    parser = add_subj_ses_arg_to(parser, "session", "ses-")
    parser.add_argument(
        '--work-dir', default=DEFAULT_WORK_DIR, type=valid_output_dir,
        help=("Path to an existing directory to store the intermediate files "
              "that this script will create. These will be deleted as the "
              "script runs unless the --keep flag is specified."
              + MSG_DEFAULT.format(DEFAULT_WORK_DIR))
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help=("Include this flag to print details as the script runs.")
    )
    return vars(parser.parse_args())


def add_subj_ses_arg_to(parser: argparse.ArgumentParser, which: str,
                        prefix: str) -> argparse.ArgumentParser:
    """
    :param parser: argparse.ArgumentParser to add participant/session ID to
    :param which: String naming the ID to add: "participant" or "session"
    :param prefix: String, the first part of a participant or session ID,
                   shared by all of them: "sub-NDARINV" or "ses-"
    :return: parser, now with a participant ID or session ID argument
    """
    abbr = prefix[:3]
    MSG_SUB_SES = ("ID of the {0} to get brain coverage values for. By "
                   "default, this script will get every {0} in the given "
                   "--bids-dir. Including this argument will restrict {0}s "
                   "to only those whose {0} ID is, or starts with, the given "
                   "argument. Wildcards are valid. Partial IDs are also "
                   "valid: for example, including --{0} {1}AA or --{0} AA "
                   "will restrict the script to {0} IDs starting with {1}AA.")
    parser.add_argument(
        f"-{abbr}", f"-{abbr}-ID", f"--{which}", f"--{which}-ID", dest=which,
        type=lambda in_arg: valid_subj_ses(in_arg, prefix, which),
        help=MSG_SUB_SES.format(which, prefix), default=prefix
    )
    return parser


class LazyDict(dict):
    """
    Dictionary subclass that can get/set items...
    ...as object-attributes: self.item is self['item']. Benefit: You can
       get/set items using '.' OR using variable names in brackets.
    ...and ignore the 'default=' code until it's needed, ONLY evaluating it
       after failing to get/set an existing key. Benefit: The 'default='
       code does not need to be valid if self already has the key.
    Extended version of LazyButHonestDict from stackoverflow.com/q/17532929
    Does not change core functionality of the Python dict type.
    TODO: Right now, trying to overwrite a LazyDict method or a core dict
          attribute will silently fail: the new value can be accessed through
          dict methods but not as an attribute. Maybe worth fixing eventually?
    """
    def __getattr__(self, __name: str):
        """
        For convenience, access items as object attributes.
        :param __name: String naming this instance's item/attribute to return
        :return: Object (any) mapped to __name in this instance
        """
        return self.__getitem__(__name)
    
    def __setattr__(self, __name: str, __value) -> None:
        """
        For convenience, set items as object attributes.
        :param __name: String, the key to map __value to in this instance
        :param __value: Object (any) to store in this instance
        """
        self.__setitem__(__name, __value)

    def lazyget(self, key, get_if_absent=lambda: None):
        """
        LazyButHonestDict.lazyget from stackoverflow.com/q/17532929
        :param key: Object (hashable) to use as a dict key
        :param get_if_absent: function that returns the default value
        :return: _type_, _description_
        """
        return self[key] if key in self else get_if_absent()
    
    def lazysetdefault(self, key, set_if_absent=lambda: None):
        """
        LazyButHonestDict.lazysetdefault from stackoverflow.com/q/17532929 
        :param key: Object (hashable) to use as a dict key
        :param set_if_absent: function that returns the default value
        :return: _type_, _description_
        """
        return (self[key] if key in self else
                self.setdefault(key, set_if_absent()))


def set_tsvrowtitle_col(task_covg_row: pd.Series) -> str:
    return f"bc_{task_covg_row['task']}_run-{task_covg_row['run']:02d}"


def get_df_from_tier1(tier1_fpath: str, COLS: LazyDict) -> pd.DataFrame:
    # Get all files with .nii.gz extension in the func folder
    tier1_df = pd.DataFrame({COLS.func[0]: glob(tier1_fpath)},
                            columns=COLS.func)
    tier1_df[COLS.func[1]] = tier1_df[COLS.func[0]].apply(os.path.basename)
    tier1_df[COLS.func[2:]] = \
        tier1_df[COLS.func[1]].str.split("_", 4).values.tolist()
    tier1_df[COLS.task_run] = \
        tier1_df[COLS.task_run].apply(split_off_relevant_value_from)
    tier1_df[COLS.task_run[1]] = tier1_df[COLS.task_run[1]].astype(int)
    tier1_df[COLS.tsv] = tier1_df.apply(set_tsvrowtitle_col, axis=1)
    return tier1_df


def get_col_names(tsv: pd.DataFrame) -> LazyDict:
    # Set dataframe column names
    COLS = LazyDict(ID=LazyDict(sub="participant_id", ses="session_id"),
                    to_get="coverage", final=tsv.columns.values.tolist())
    COLS.ID.sub_ses = [COLS.ID.sub, COLS.ID.ses]
    COLS.task_run = ["task", "run"]
    COLS.func = ["fpath", "base_name", *COLS.ID.sub_ses, *COLS.task_run, "ext"]
    COLS.func_non_ID = set(COLS.func) - set(COLS.ID.sub_ses)
    COLS.headers = tsv.columns[tsv.columns.str.startswith("bc_")
                               ].values.tolist()
    COLS.tsv = "tsv_col"
    return COLS


def split_off_relevant_value_from(col: pd.Series) -> pd.Series:
    df = col.to_frame()
    df[["irrelevant", "relevant"]] = col.str.split("-").values.tolist()
    return df.pop("relevant")


def calculate_coverage_and_add_to(tier1_df: pd.DataFrame, cli_args: dict,
                                  COLS: LazyDict) -> pd.DataFrame:
    
    if cli_args["verbose"]:
        counts = tier1_df.nunique()
        print(f"Getting {COLS.to_get} for {counts.run} run(s) of {counts.task} "
              f"task(s) of {counts.session_id} session(s) of "
              f"{counts.participant_id} subject(s).")
    tier1_df[COLS.to_get] = tier1_df.apply(
        lambda row: get_coverage_1_row(row, cli_args), axis=1
    )

    if cli_args["verbose"]:
        print(tier1_df)
    return tier1_df


def update_participants_tsv(tsv: pd.DataFrame, tier1_df: pd.DataFrame,
                            cli_args: dict, COLS: LazyDict) -> None:
    if cli_args["verbose"]:
        print(tsv)
    new_tsv = transform_1_df_to_BIDS_DB(tier1_df, COLS)
    if cli_args["verbose"]:
        print(new_tsv)
    final_tsv = tsv.drop(columns=COLS.headers).merge(new_tsv)[COLS.final]
    with open(cli_args["out_tsv"], "a", newline="") as outfile:
        final_tsv.drop_duplicates().to_csv(outfile, sep='\t',
                                           index=False, mode="a",
                                           header=(not outfile.tell()))
            

def make_subj_ses_dict(a_df: pd.DataFrame, COLS: LazyDict) -> dict:  # sub_ses_cols, col_to_get: str, dtype: str
    sub_ses_dict = {col: None for col in COLS.func}
    if not a_df.empty:
        for col_ID in COLS.ID.sub_ses:
            sub_ses_dict[col_ID] = a_df[col_ID].iloc[0]
        def add_to_sub_ses_dict(row):
            sub_ses_dict[row.get(COLS.tsv)] = row.get(COLS.to_get)
        a_df.apply(add_to_sub_ses_dict, axis=1)
    return sub_ses_dict


def transform_1_df_to_BIDS_DB(df: pd.DataFrame, COLS: LazyDict
                              ) -> pd.DataFrame: # sub_ses_cols, col_to_get: str, dtype: str
    sub_ses_rows = df.groupby(COLS.ID.sub_ses).apply(
        lambda sub_ses_df: make_subj_ses_dict(sub_ses_df, COLS)
    )
    return pd.DataFrame(sub_ses_rows.values.tolist(),
                        columns=[*COLS.ID.sub_ses, *COLS.headers]
                        ).drop_duplicates()


def get_coverage_1_row(task_covg_row: pd.Series, cli_args: dict) -> float:
    coverage = None
    path_to = get_image_paths(task_covg_row.base_name, cli_args)
    os.makedirs(path_to.temp_dir, exist_ok=True)

    # FSL calls to perform necessary calculations 
    # Changes image to float type 
    # print(f"Starting calculations for image:\n{task_covg_row.fpath}")
    MathsCommand(in_file=task_covg_row.fpath, out_file=path_to.prefiltered,
                 output_datatype="float", output_type="NIFTI_GZ").run()
    
    # If calculation fails to make prefiltered image, then move on
    if not os.path.exists(path_to.prefiltered):
        if cli_args["verbose"]:
            print(f"'{path_to.prefiltered}' is not an existing path. A RUN "
                  F"FROM THIS SUBJECT FAILED: {task_covg_row.participant_id}")
        with open(cli_args["failed_file"], "a") as file:
            file.write(f"This image failed: '{path_to.prefiltered}'")
    else:
        # Calculate brain coverage percent & put it in column BC_task_run-# 
        coverage = calculate_coverage_percent(path_to)
        
        # If keep flag not specified, remove intermediate files
        if not cli_args["keep"]:
            shutil.rmtree(path_to.temp_dir)
    return coverage


def calculate_coverage_percent(path_to: LazyDict) -> float:
    """
    :param path_to: LazyDict mapping "prefiltered", "bold", "mask",
                    "mask_masked", and "MNI" to .nii.gz file paths, and
                    "temp_dir" to a working directory path
    :return: Float between 0 and 100, brain coverage percentage
    """
    # Takes mean of T dimension
    fsl.MeanImage(in_file=path_to.prefiltered, output_type="NIFTI_GZ",
                  dimension="T", out_file=path_to.bold).run()
    fsl.UnaryMaths(in_file=path_to.bold, output_type="NIFTI_GZ",
                   operation="bin", out_file=path_to.mask).run()

    # Apply MNI mask to binned image 
    fsl.ApplyMask(in_file=path_to.mask, output_type="NIFTI_GZ",
                  mask_file=path_to.MNI, out_file=path_to.mask_masked).run()

    # Loads final masked image and MNI mask and counts their voxels
    n_vox = {img: float(imagestats.count_nonzero_voxels(
                    nib.load(path_to[img])
             )) for img in ("mask_masked", "MNI")}

    # Calculate brain coverage percent & put it in column BC_task_run-# 
    return round((n_vox["mask_masked"]/n_vox["MNI"]) * 100, 3)


def get_image_paths(base_name: str, cli_args: dict) -> dict:
    """
    Get a dict of paths to images that will be created
    :param base_name: String, a BIDS-valid .nii.gz file base name
    :param cli_args: _type_, _description_
    :return: Dict mapping "prefiltered", "bold", "mask", "mask_masked", "MNI",
             to .nii.gz file paths, and "temp_dir" to a working directory path
    """
    path_to = LazyDict({"MNI": cli_args["mni_template"], "temp_dir":
                        os.path.join(cli_args["work_dir"], base_name)})
    exclude = {"mean", "func"}  # Exclude redundant words from key names
    for which in ("prefiltered_func", "bold_mean_func",
                  "mean_func_mask", "mean_func_mask_masked"):
        key = "_".join(wd for wd in which.split("_") if wd not in exclude)
        path_to[key] = os.path.join(path_to.temp_dir,
                                    f"{base_name}_{which}.nii.gz")
    return path_to


def get_and_print_time_if(will_print: bool, event_time: datetime,
                          event_name: str) -> datetime:
    """
    Print and return a string showing how much time has passed since the
    current running script reached a certain part of its process
    :param will_print: True to print an easily human-readable message
                       showing how much time has passed since {event_time}
                       when {event_name} happened, False to skip printing
    :param event_time: datetime object representing a time in the past
    :param event_name: String to print after 'Time elapsed '
    :return: datetime object representing the current moment
    """
    timestamp = datetime.now()
    if will_print:
        print(f"\nTime elapsed {event_name}: {timestamp - event_time}")
    return timestamp


def valid_output_dir(path) -> str:
    """
    Try to make a folder for new files at path; throw exception if that fails
    :param path: String which is a valid (not necessarily real) folder path
    :return: String which is a validated absolute path to real writeable folder
    """
    return validate(path, lambda x: os.access(x, os.W_OK),
                    valid_readable_dir, "Cannot create directory at '{}'", 
                    lambda y: os.makedirs(y, exist_ok=True))


def valid_readable_dir(path) -> str:
    """
    :param path: Parameter to check if it represents a valid directory path
    :return: String representing a valid directory path
    """
    return validate(path, os.path.isdir, valid_readable_file,
                    "Cannot read directory at '{}'")


def valid_readable_file(path) -> str:
    """
    Throw exception unless parameter is a valid readable filepath string. Use
    this, not argparse.FileType('r') which leaves an open file handle.
    :param path: Parameter to check if it represents a valid filepath
    :return: String representing a valid filepath
    """
    return validate(path, lambda x: os.access(x, os.R_OK),
                    os.path.abspath, "Cannot read file at '{}'")


def valid_subj_ses(in_arg, prefix: str, name: str) -> str:
    """
    :param in_arg: Object to check if it is a valid subject ID or session name
    :param prefix: String, 'sub-' or 'ses-'
    :param name: String describing what in_arg should be (e.g. 'subject')
    :return: True if in_arg is a valid subject ID or session name; else False
    """
    return validate(in_arg, lambda _: True,
                    lambda y: (y if y[:len(prefix)] == prefix else prefix + y),
                    f"{{}} is not a valid {name}")


def validate(to_validate, is_real, make_valid, err_msg: str, prepare=None):
    """
    Parent/base function used by different type validation functions. Raises an
    argparse.ArgumentTypeError if the input object is somehow invalid.
    :param to_validate: String to check if it represents a valid object 
    :param is_real: Function which returns true iff to_validate is real
    :param make_valid: Function which returns a fully validated object
    :param err_msg: String to show to user to tell them what is invalid
    :param prepare: Function to run before validation
    :return: to_validate, but fully validated
    """
    try:
        if prepare:
            prepare(to_validate)
        assert is_real(to_validate)
        return make_valid(to_validate)
    except (OSError, TypeError, AssertionError, ValueError, 
            argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError(err_msg.format(to_validate))


def main():
    '''Study input should be overarching project folder, which contains derivative and template subfolders'''
    cli_args = _cli()
    start_time = datetime.now()
    if cli_args["verbose"]:
        print(f"Started at {start_time}")

    # Load the existing participants.tsv into a DataFrame
    tsv = pd.read_csv(cli_args["in_tsv"], sep='\t')
    tsv.drop_duplicates(inplace=True)

    # Fix run numbers in column headers in participants.tsv
    tsv.rename(columns={f'bc_{task}_run-{i}': f'bc_{task}_run-0{i}' for i in
                        range(10) for task in ('nback', 'MID', 'SST', 'rest')
                        }, inplace=True)

    COLS = get_col_names(tsv)

    # Collect tier1 file paths and put them into df
    subj_glob = cli_args["participant"] + "*"
    ses_glob = cli_args["session"] + "*"
    tier1_fpath = os.path.join(
        cli_args["bids_dir"], subj_glob, ses_glob, "func",
        f"{subj_glob}_{ses_glob}_task-*_run-*_bold.nii.gz"
    )
    tier1_df = get_df_from_tier1(tier1_fpath, COLS)

    # Verify that participants.tsv and tier1 df have this subject session(s)
    tsv = tsv[tsv["participant_id"].str.startswith(cli_args["participant"])
              ].merge(tier1_df, on=COLS.ID.sub_ses)
    if tsv.empty:
        sys.exit(f"Error: No '{tier1_fpath}' files are for participants who "
                 f"have sessions in '{cli_args['in_tsv']}'")
    # else:

    tier1_df = calculate_coverage_and_add_to(tier1_df, cli_args, COLS)
    get_and_print_time_if(cli_args["verbose"], start_time,
                          "calculating task coverage")

    update_participants_tsv(tsv, tier1_df, cli_args, COLS)
    get_and_print_time_if(cli_args["verbose"], start_time,
                          f"since {sys.argv[0]} started running")


if __name__ == "__main__":
    main()
