"""Data loading, trial labelling and statistics for SequenceFingerSpeech version 2 (SFS2)."""

import json
from pathlib import Path

import pandas as pd
import pingouin as pg

repo_root = Path(__file__).resolve().parent

# The data folder is looked up locally first, then on the lab server.
_data_root_candidates = [
    repo_root / "SFS2",
    Path(r"Y:\data\SequenceFingerSpeech\raw\SFS2"),
]
data_root = next((p for p in _data_root_candidates if p.exists()), None)
if data_root is None:
    raise FileNotFoundError(
        "SFS2 data folder not found. Put it at Version2/SFS2 or mount the lab server."
    )
dat_prefix = data_root / "SequenceFingerSpeech"


# ----------------------------------------------------------------------------
# Experiment design
# ----------------------------------------------------------------------------

seq_length = 11

# Effector codes used in the .dat files.
EFFECTORS = {0: "speech", 2: "hand"}

# Each subject's group indexes a 6-sequence assignment written by TargetFileMaker.ipynb:
#   positions 0, 2 -> speech-trained; 1, 3 -> hand-trained; 4, 5 -> untrained.
_assignments_path = repo_root / "group_sequence_assignments.json"
if not _assignments_path.exists():
    raise FileNotFoundError(
        f"Missing {_assignments_path}. Run TargetFileMaker.ipynb first to create balanced group assignments."
    )
group_assignments = [tuple(p) for p in json.loads(_assignments_path.read_text())]
SPEECH_TRAINED_POS = (0, 2)
HAND_TRAINED_POS = (1, 3)

# Block layout (BN). Block 1 is familiarization; each day is pre-test, training, post-test.
FAMILIARIZATION_BLOCKS = [1]
DAY_BLOCKS = {1: range(1, 14), 2: range(14, 26)}
PRE_TEST_BLOCKS = [2, 3, 4, 14, 15, 16]
POST_TEST_BLOCKS = [11, 12, 13, 23, 24, 25]

PHASE_ORDER = ["pre-test", "train", "post-test"]
CONDITION_ORDER = ["trained", "transfer", "untrained"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def read_dat_file(path) -> pd.DataFrame:
    """Read one subject's .dat file. All columns are int except the cued sequence."""
    data = pd.read_csv(path, sep="\t", usecols=lambda c: not c.startswith("Unnamed"), index_col=False)
    for col in data.columns:
        data[col] = data[col].astype(str if col == "cue" else int)
    return data


def read_subject(sub: int) -> pd.DataFrame:
    return read_dat_file(f"{dat_prefix}_{sub}.dat")


def add_IPI(subj: pd.DataFrame) -> pd.DataFrame:
    """Add inter-press intervals IPI1..IPI10 (hand trials only carry press times)."""
    for i in range(1, seq_length):
        subj[f"IPI{i}"] = subj[f"pressTime{i + 1}"] - subj[f"pressTime{i}"]
    return subj


def sequence_condition(group: int, cue: str, effector: int) -> str:
    """Whether `cue` is trained, transfer or untrained when produced with `effector`.

    A sequence trained with the other effector counts as transfer.
    """
    pos = group_assignments[group].index(cue)
    if pos in SPEECH_TRAINED_POS:
        return "trained" if effector == 0 else "transfer"
    if pos in HAND_TRAINED_POS:
        return "trained" if effector == 2 else "transfer"
    return "untrained"


def block_phase(bn: int) -> str:
    if bn in PRE_TEST_BLOCKS:
        return "pre-test"
    if bn in POST_TEST_BLOCKS:
        return "post-test"
    return "train"


def block_day(bn: int) -> int:
    return next(day for day, blocks in DAY_BLOCKS.items() if bn in blocks)


def load_trials(subjects) -> pd.DataFrame:
    """Load all subjects' trials, drop familiarization, and label every trial.

    Added columns: `effector_name`, `condition` (trained/transfer/untrained),
    `phase` (pre-test/train/post-test) and `day`.
    """
    trials = pd.concat([add_IPI(read_subject(s)) for s in subjects], ignore_index=True)
    trials = trials[~trials["BN"].isin(FAMILIARIZATION_BLOCKS)].reset_index(drop=True)

    trials["effector_name"] = trials["effector"].map(EFFECTORS)
    trials["condition"] = [
        sequence_condition(g, c, e) for g, c, e in zip(trials["group"], trials["cue"], trials["effector"])
    ]
    trials["phase"] = trials["BN"].map(block_phase)
    trials["day"] = trials["BN"].map(block_day)
    return trials


# ----------------------------------------------------------------------------
# Summaries and statistics
# ----------------------------------------------------------------------------

def per_block(trials: pd.DataFrame, effector: str, measure: str = "ET") -> pd.DataFrame:
    """Median `measure` of correct trials per subject x block x condition."""
    correct = trials[(trials["effector_name"] == effector) & (trials["isError"] == 0)]
    return (
        correct.groupby(["SubNum", "day", "phase", "BN", "condition"])[measure]
        .median()
        .reset_index()
    )


def error_rate_per_block(trials: pd.DataFrame, effector: str) -> pd.DataFrame:
    """Proportion of error trials per subject x block x condition."""
    eff = trials[trials["effector_name"] == effector]
    return (
        eff.groupby(["SubNum", "day", "phase", "BN", "condition"])["isError"]
        .mean()
        .rename("error_rate")
        .reset_index()
    )


def per_test_phase(trials: pd.DataFrame, effector: str, measure: str = "ET") -> pd.DataFrame:
    """Median `measure` of correct trials per subject x day x test phase x condition.

    Trials are pooled directly within each test phase (all 3 blocks together),
    not first summarised per block and then averaged across blocks.
    """
    correct = trials[
        (trials["effector_name"] == effector)
        & (trials["isError"] == 0)
        & (trials["phase"] != "train")
    ]
    return (
        correct.groupby(["SubNum", "day", "phase", "condition"])[measure]
        .median()
        .reset_index()
    )


def learning_tests(prepost: pd.DataFrame, measure: str = "ET") -> pd.DataFrame:
    """Paired t-tests on the change from pre- to post-test of each learning contrast.

    Contrasts are taken against untrained sequences:
      specific = trained - untrained, transfer = transfer - untrained.
    With a time measure, a negative post - pre change means that sequence got faster
    than untrained. The 'day 1 pre -> day 2 post' rows span the whole experiment.
    """
    wide = prepost.pivot_table(index=["SubNum", "day", "phase"], columns="condition", values=measure)
    contrasts = {
        "specific": wide["trained"] - wide["untrained"],
        "transfer": wide["transfer"] - wide["untrained"],
    }
    windows = [
        ("day 1", (1, "pre-test"), (1, "post-test")),
        ("day 2", (2, "pre-test"), (2, "post-test")),
        ("day 1 pre -> day 2 post", (1, "pre-test"), (2, "post-test")),
    ]

    rows = []
    for window, (pre_day, pre_phase), (post_day, post_phase) in windows:
        for name, contrast in contrasts.items():
            pre = contrast.xs((pre_day, pre_phase), level=["day", "phase"])
            post = contrast.xs((post_day, post_phase), level=["day", "phase"])
            paired = pd.concat({"pre": pre, "post": post}, axis=1).dropna()
            test = pg.ttest(paired["post"], paired["pre"], paired=True).iloc[0]
            rows.append({
                "window": window,
                "learning": name,
                "n": len(paired),
                "mean_pre": paired["pre"].mean(),
                "mean_post": paired["post"].mean(),
                "post_minus_pre": (paired["post"] - paired["pre"]).mean(),
                "T": test["T"],
                "dof": test["dof"],
                "p": test["p_val"],
                "cohen_d": test["cohen_d"],
            })
    return pd.DataFrame(rows)
