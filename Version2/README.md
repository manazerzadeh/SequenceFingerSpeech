# Version 2 (SFS2)

## Design

Each subject gets 6 sequences: 2 trained with speech, 2 trained with the right hand, 2 untrained. Role assignments are balanced across subjects and stored in `group_sequence_assignments.json` (subject group *g* uses row *g*; positions 0, 2 are speech-trained, 1, 3 hand-trained, 4, 5 untrained).

| Blocks | Phase | Trials per block |
|--------|-------|------------------|
| 1 | familiarization (random sequences, excluded from analysis) | 20 |
| 2–4, 14–16 | pre-test (day 1, day 2) | 36: all 6 sequences × both effectors × 3 |
| 5–10, 17–22 | training | 32: 4 trained sequences in their effector × 8 |
| 11–13, 23–25 | post-test | 36 |

Effector codes in the `.dat` files: 0 = speech, 2 = right hand.

## Files

| File | Purpose |
|------|---------|
| `Analysis.ipynb` | Main analysis: block time courses, pre/post-tests and learning statistics for speech and hand |
| `TargetFileMaker.ipynb` | Builds the balanced group assignments and the `.tgt` target files for the experiment |
| `utils.py` | Data loading, trial labelling (condition, phase, day), summaries and t-tests |
| `plotting.py` | Figure style and plotting functions |
| `group_sequence_assignments.json` | Sequence roles per group, written by `TargetFileMaker.ipynb` and read by `utils.py` |

`SFS2/` (raw data) and `TargetFiles/` are gitignored. Re-running `TargetFileMaker.ipynb` regenerates the same target files from a fixed seed.

## Analysis notes

- Measure: execution time `ET` of correct trials.
- Pre/post-test values are the median over all trials of a test phase (its 3 blocks pooled) for each subject and condition.
- Learning tests: paired t-tests on the post − pre change of trained − untrained (specific) and transfer − untrained (transfer), for day 1, day 2 and day 1 pre → day 2 post.
