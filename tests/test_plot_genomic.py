import pandas as pd
import matplotlib.pyplot as plt
from methlevels.plot_genomic import (
    add_label_positions_to_intervals,
    compute_row_placement_for_vertical_interval_dodge,
)


def test_add_label_positions_to_intervals():

    itvls = pd.DataFrame(
        dict(
            Start=[10, 20, 30],
            End=[15, 25, 35],
            name=["aasdfasdfasdf", "b", "casdfasdfasdfasdf"],
        )
    )
    xlim = (0, 40)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=180, figsize=(2, 2))
    fig.set_constrained_layout_pads(hspace=0, wspace=0, h_pad=0, w_pad=0)
    ax.set_xlim(xlim)

    itvls_w_labels = add_label_positions_to_intervals(
        itvls=itvls,
        ax=ax,
        xlim=xlim,
        fontsize=6,
    )


def test_dodge_labeled_intervals_vertically():

    itvls = pd.DataFrame(
        dict(
            Start=[10, 20, 30],
            End=[15, 25, 35],
            name=["aasdfasdfasdf", "b", "casdfasdfasdfasdf"],
        )
    )
    xlim = (0, 40)

    fig, ax = plt.subplots(1, 1, constrained_layout=True, dpi=180, figsize=(2, 2))
    fig.set_constrained_layout_pads(hspace=0, wspace=0, h_pad=0, w_pad=0)
    ax.set_xlim(xlim)

    itvls_w_labels = add_label_positions_to_intervals(
        itvls=itvls,
        ax=ax,
        xlim=xlim,
        fontsize=6,
    )

    row_ser = compute_row_placement_for_vertical_interval_dodge(intervals=itvls_w_labels)

    # try with string index (instead of int index above)
    itvls_w_labels.index = ['a', 'b', 'c']

    row_ser = compute_row_placement_for_vertical_interval_dodge(intervals=itvls_w_labels)
