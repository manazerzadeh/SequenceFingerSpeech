"""Figure style and plots for SequenceFingerSpeech version 2 (SFS2)."""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

import utils

CONDITION_LABELS = {"trained": "Trained", "transfer": "Transfer", "untrained": "Untrained"}
EFFECTOR_TITLES = {"speech": "Speech", "hand": "Right hand"}


def set_figure_style(scale="1col"):
    """Set rcParams for publication figures.

    scale: "1col" (7.62 cm wide), "1.5col" (11.6 cm) or "2col" (16.5 cm); height follows the golden ratio.
    """
    widths_cm = {"1col": 7.62, "1.5col": 11.6, "2col": 16.5}
    if scale not in widths_cm:
        raise ValueError("Invalid scale. Choose from '1col', '1.5col', or '2col'.")

    width_in = widths_cm[scale] / 2.54
    golden_ratio = (5**0.5 - 1) / 2
    rcParams["figure.figsize"] = (width_in, width_in * golden_ratio)

    rcParams["font.size"] = 10
    rcParams["axes.titlesize"] = 12
    rcParams["axes.labelsize"] = 9
    rcParams["xtick.labelsize"] = 7
    rcParams["ytick.labelsize"] = 7
    rcParams["legend.fontsize"] = 8
    rcParams["figure.titleweight"] = "bold"

    rcParams["axes.linewidth"] = 0.75
    rcParams["xtick.major.width"] = 0.75
    rcParams["ytick.major.width"] = 0.75

    rcParams["text.usetex"] = False
    rcParams["axes.formatter.use_mathtext"] = True


def add_subpanel_label(ax, label, fontsize=20, position=(-0.1, 1.05)):
    """Add a bold subpanel label (e.g. 'a') to the top-left of `ax`."""
    ax.text(position[0], position[1], label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="top", ha="left")


def _relabel_legend(ax, title="Condition"):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [CONDITION_LABELS.get(l, l) for l in labels], title=title, frameon=False)


def _mark_block_boundaries(ax, blocks):
    """Dashed lines between phases (grey) and between days (black) on a categorical block axis."""
    for i in range(1, len(blocks)):
        prev, curr = blocks[i - 1], blocks[i]
        if utils.block_day(prev) != utils.block_day(curr):
            ax.axvline(i - 0.5, color="black", linestyle="--", linewidth=1.75)
        elif utils.block_phase(prev) != utils.block_phase(curr):
            ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=0.75)


def plot_blocks(per_block, y, effector, ylabel=None, ax=None):
    """Point plot of `y` across blocks, one line per condition (mean +/- SE across subjects)."""
    if ax is None:
        _, ax = plt.subplots()
    blocks = sorted(per_block["BN"].unique())
    sns.pointplot(
        data=per_block, x="BN", y=y, order=blocks,
        hue="condition", hue_order=utils.CONDITION_ORDER,
        dodge=True, errorbar="se", ax=ax,
    )
    _mark_block_boundaries(ax, blocks)
    _relabel_legend(ax)
    ax.set(xlabel="Block", ylabel=ylabel or y, title=f"{EFFECTOR_TITLES[effector]} performance")
    sns.despine(ax=ax, trim=True)
    return ax


def plot_prepost(prepost, y, effector, ylabel=None):
    """Pre- vs post-test per condition, one panel per day (mean +/- SE across subjects)."""
    g = sns.catplot(
        data=prepost, kind="point",
        x="phase", y=y, order=["pre-test", "post-test"],
        hue="condition", hue_order=utils.CONDITION_ORDER,
        col="day", dodge=True, errorbar="se",
        height=5, aspect=0.9,
    )
    g.set_axis_labels("", ylabel or y)
    g.set_titles("Day {col_name}")
    for text in g.legend.texts:
        text.set_text(CONDITION_LABELS.get(text.get_text(), text.get_text()))
    g.legend.set_title("Condition")
    g.figure.suptitle(f"{EFFECTOR_TITLES[effector]} performance", y=1.05)
    return g
