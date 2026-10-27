# SequenceFingerSpeech

Does learning a sequence with one effector transfer to another? Subjects practise 11-element sequences (digits 1–5) either by **speaking** them or by **pressing** them with the right hand. Pre- and post-tests measure performance on every sequence in both effectors. If training in one effector speeds up the same sequence in the other, that points to an abstract sequence representation shared across effectors.

For each effector, sequences fall into three conditions:

| Condition | Meaning |
|-----------|---------|
| trained   | practised with this effector |
| transfer  | practised with the *other* effector |
| untrained | never practised |

Specific learning is trained − untrained and transfer is transfer − untrained, each compared from pre- to post-test.

## Versions

| Folder | Design | Status |
|--------|--------|--------|
| [Version1](Version1/) | 4 sequences: 1 trained with speech, 1 with the hand, 2 untrained (2 counterbalanced groups). One session. | Done (n = 7). Specific learning in both effectors; no transfer (transfer ≈ untrained). |
| [Version2](Version2/) | 6 sequences: 2 trained with speech, 2 with the hand, 2 untrained. Two days, balanced assignments across subjects. | Collecting data. |

Version2 trains two sequences per effector because single-sequence training can behave atypically, which may have hidden transfer in Version1.

## Setup

```bash
pip install -r requirements.txt
```

Raw data is not in the repository. Put it in the version's folder (`Version1/SFS1/`, `Version2/SFS2/`) or mount the lab server (`Y:\data\SequenceFingerSpeech\raw\`). Run the notebooks from inside their version folder so `import utils` resolves.
