import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
from scipy import stats
import matplotlib.cm as cm
import seaborn as sns
from typing import List
import pingouin as pg

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import itertools

from pathlib import Path
import os


from natsort import index_natsorted

# path = "./SFS2/SequenceFingerSpeech"
# path_misc = "./SFS2_miscs/"

# path = r"Y:\data\SequenceFingerSpeech\raw\SFS2\SequenceFingerSpeech"

repo_root = Path(__file__).resolve().parent

# possible ROOTS (not full dataset path yet)
candidates = [
    repo_root / "SFS2",
    Path(r"Y:\data\SequenceFingerSpeech\raw\SFS2"),
]

base_root = next((p for p in candidates if p.exists()), None)

if base_root is None:
    raise FileNotFoundError("SFS2 root not found")

# now append dataset folder once
path = base_root / "SequenceFingerSpeech"
path_misc = base_root / "SFS2_miscs"

# optional: convert to string if you need it
path = str(path)
path_misc = str(path_misc)


seq_length = 11

fingers = ['1', '2', '3', '4', '5'] #mapping of fingers to numbers

sequences = ['13524232514' ,'54231251343',  '21435235214', '34352452141', '42531451342', '31525423241']
# generate all possible permutations of the 6 sequences
# while treating swaps of the last two positions as the same permutation
all_perms = itertools.permutations(sequences, 6)

unique_perms = []
seen = set()

g_sequences_seen = set()


for perm in all_perms:
    key = perm[:4] + tuple(sorted(perm[4:]))  # ignore order of last two
    if key not in seen and key not in g_sequences_seen:
        seen.add(key)
        unique_perms.append(perm)

pre_test_blocks = [2,3,4, 14, 15, 16]
post_test_blocks = [11, 12, 13, 23, 24, 25]

iti = 3000   #Inter trial interval
precue_time = 2000 #Time between precue and go cue
hand = 2 #left or right hand

def read_dat_file(path : str):
    column_names = pd.read_csv(path, delimiter='\t', usecols=lambda column: not column.startswith("Unnamed")).columns
    data = pd.read_csv(path, delimiter= '\t', usecols=lambda column: not column.startswith("Unnamed"), index_col=False)
    # change dtypes of all columns to int except cue
    for col in data.columns:
        if col != 'cue':
            data[col] = data[col].astype(int)
        if col == 'cue':
            data[col] = data[col].astype(str)

    return data

def read_dat_files_subjs_list(subjs_list: List[int]):
    """
    Reads the corresponding dat files of subjects and converts them to a list of dataframes.
    """
    return [read_dat_file(path + "_" + str(sub) + ".dat") for sub in subjs_list]


def read_dat_files_subjs_list_speech(subjs_list: List[int]):
    """
    Reads the corresponding dat files of subjects and converts them to a list of dataframes.
    """
    return [read_dat_file(path + "SequenceFingerSpeech_" + str(sub) + "_sp.dat") for sub in subjs_list]



def remove_error_trials(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Removes error trials from the dat file of a subject
    """

    return subj[(subj['isError'] == 0)]


def remove_error_trials_presses(subj_press: pd.DataFrame) -> pd.DataFrame:

    return subj_press[(subj_press['isTrialError'] == 0) & (subj_press['timingError'] == 0)]


def remove_error_presses(subj_press: pd.DataFrame) -> pd.DataFrame:

    return subj_press[(subj_press['isPressError']) == 0]



def add_IPI(subj: pd.DataFrame):
    """
    Adds interpress intervals to a subject's dataframe
    """

    for i in range(seq_length-1):
        col1 = 'pressTime'+str(i+1)
        col2 = 'pressTime'+str(i+2)
        new_col = 'IPI'+str(i+1)
        subj[new_col] = subj[col2] - subj[col1]

    # subj['IPI0'] = subj['RT']




def add_trained_transfer_untrained_flag(row: pd.Series) -> int:
    """
    Adds a flag to each row of the dataframe indicating whether the sequence is trained or untrained for the subject
    """
    group = row['group']
    seq = row['cue']
    effector = row['effector']
    # get the index of the sequence in the group sequences
    # seq_index = g_sequences[group].index(seq)
    seq_index = unique_perms[group].index(seq)
    if seq_index in ([0, 2]):
        if effector == 0:
            return 'trained'
        else:
            return 'transfer'
    elif seq_index in ([1, 3]):
        if effector == 0:
            return 'transfer'
        else:
            return 'trained'
    else:
        return 'untrained'
    

def add_pre_train_post_flag(row: pd.Series) -> int:
    """
    Adds a flag to each row of the dataframe indicating whether the trial is pre-test, train or post-test for the subject
    """
    if row['BN'] in pre_test_blocks:
        return 'pre-test'
    elif row['BN'] in post_test_blocks:
        return 'post-test'
    else:
        return 'train'


def finger_melt_IPIs(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Creates seperate row for each IPI in the whole experiment adding two columns, "IPI_Number" determining the order of IPI
    and "IPI_Value" determining the time of IPI
    """

    
    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'repType', 'repNum', 'seqNum', 'dummy', 
                             'tStart', 'hand', 'cueP', 'RT', 'MT', 'points',
                            'isError'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('IPI')],
                    var_name='IPI_Number', 
                    value_name='IPI_Value')
    

    subj_melted['N'] = (subj_melted['IPI_Number'].str.extract('(\d+)').astype('int64') + 1)

    

    
    return subj_melted


def finger_melt_presses(subj: pd.DataFrame) -> pd.DataFrame:

    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'repType', 'repNum', 'seqNum', 'dummy', 
                             'tStart', 'hand', 'cueP', 'RT', 'MT', 'points',
                            'isError'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('press') and not _.startswith('pressTime')],
                    var_name='Press_Number', 
                    value_name='Press_Value')
    

    subj_melted['N'] = subj_melted['Press_Number'].str.extract('(\d+)').astype('int64')

    return subj_melted


def finger_melt_responses(subj: pd.DataFrame) -> pd.DataFrame:

    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'repType', 'repNum', 'seqNum', 'dummy', 
                             'tStart', 'hand', 'cueP', 'RT', 'MT', 'points',
                            'isError'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('response')],
                    var_name='Response_Number', 
                    value_name='Response_Value')
    
    subj_melted['N'] = subj_melted['Response_Number'].str.extract('(\d+)').astype('int64')

    return subj_melted


def finger_melt(subj: pd.DataFrame) -> pd.DataFrame:
    melt_IPIs = finger_melt_IPIs(subj)
    melt_presses = finger_melt_presses(subj)
    melt_responses = finger_melt_responses(subj)
    merged_df = melt_IPIs.merge(melt_presses, on = ['BN', 'TN', 'SubNum', 'repType', 'repNum', 'seqNum', 'dummy', 
                             'tStart', 'hand', 'cueP', 'RT', 'MT', 'points',
                            'isError','N'])\
                                               .merge(melt_responses, on = ['BN', 'TN', 'SubNum', 'repType', 'repNum', 'seqNum', 'dummy', 
                             'tStart', 'hand', 'cueP', 'RT', 'MT', 'points',
                            'isError', 'N'] )

    return add_press_error(merged_df)


def add_press_error(merged_df):
    merged_df['isPressError'] = ~(merged_df['Press_Value'] == merged_df['Response_Value'])
    return merged_df



def finger_melt_Forces(subjs_force: pd.DataFrame) -> pd.DataFrame:
    """
    Creates seperate row for each Finger Force in the whole experiment adding two columns, "Force_Number" determining the order of Force
    and "Force_Value" determining the time of Force
    """

    
    subj_force_melted = pd.melt(subjs_force, 
                    id_vars=['state', 'time', 'BN', 'TN', 'SubNum', 'group', 'isTrain', 'isClamped', 
                            'RT', 'ET'
                            ,'speed',
                            'isError', 'isCross', 'isPresshard'],
                    value_vars =  [_ for _ in subjs_force.columns if _.startswith('force')],
                    var_name='Force_Number', 
                    value_name='Force_Value')
    
    return subj_force_melted


def cut_force(subjs_force: pd.DataFrame, side_padding) -> pd.DataFrame:
    """
    Cuts the force data to the same length as the IPI data
    """
    subjs_force = subjs_force[((subjs_force['RT'] + precue_time) <= (subjs_force['time'] + side_padding)) & 
                              ((subjs_force['time']) <= (subjs_force['ET'] + precue_time + side_padding))]
    return subjs_force



def cut_force_left(subjs_force: pd.DataFrame) -> pd.DataFrame:

    subjs_force = subjs_force[((subjs_force['RT'] + precue_time) >= subjs_force['time'])]
    return subjs_force


def cut_force_right(subjs_force: pd.DataFrame) -> pd.DataFrame:

    subjs_force = subjs_force[((subjs_force['ET'] + precue_time) <= subjs_force['time'])]
    return subjs_force

