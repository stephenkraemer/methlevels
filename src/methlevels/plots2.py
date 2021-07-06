from copy import copy
from typing import Optional, List, Union, Dict, Tuple

import matplotlib.text as mpltext
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.patches as patches

import numpy as np
import pandas as pd

idxs = pd.IndexSlice
import scipy.interpolate as interpolate
import scipy.stats
from astropy.convolution import convolve
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import medfilt
import pyranges as pr

from methlevels import MethStats
from methlevels.utils import NamedColumnsSlice as ncls

import matplotlib.patches as mpatches
import matplotlib.collections


def plot_gene_model(df, ax):
    """
    - no genes, only transcripts with parts: intron, exon, UTR
    """

    # %%
    n_utr_and_exons_features_expected = (
        df.query('feature == "exon" or "UTR" in feature')
        .drop_duplicates(subset=["Chromosome", "Start", "End", "transcript_id"])
        .shape[0]
    )

    transcript_parts_dfs = []
    for _unused, group_df in df.groupby("transcript_id"):
        gr_exons = pr.PyRanges(group_df.query('feature == "exon"'))
        gr_utrs = pr.PyRanges(group_df.query('"UTR" in feature'))
        gr_exons = gr_exons.subtract(gr_utrs)
        transcript_parts_dfs.append(pd.concat([gr_exons.df, gr_utrs.df], axis=0))

    features = (
        pd.concat(transcript_parts_dfs, axis=0)
        .groupby("transcript_id")
        .apply(lambda df: df.sort_values(["Start", "End"]))
        .droplevel(-1)
    )

    assert n_utr_and_exons_features_expected == features.shape[0]
    # %%

    transcripts_sorted_by_start_and_length = (
        df.query('feature == "transcript"')
        .assign(length_neg=lambda df: -(df.End - df.Start))
        .sort_values(["Start", "length_neg"])
        .set_index("transcript_id")
    )

    transcripts_sorted_by_start_and_length[
        "gene_name_text_width_data_coords"
    ] = transcripts_sorted_by_start_and_length["gene_name"].apply(
        get_text_width_data_coordinates, ax=ax
    )
    transcripts_sorted_by_start_and_length["gene_name_text_width_data_coords"]
    # %%

    # %%
    # MUST be done before determining label sizes (?)
    # ax.clear()
    xmin = transcripts_sorted_by_start_and_length.Start.min()
    xmax = transcripts_sorted_by_start_and_length.End.max()
    ax.set_xlim(xmin, xmax)

    t = transcripts_sorted_by_start_and_length
    t["label_size_data_coords"] = t.gene_name.apply(
        get_text_width_data_coordinates, ax=ax
    )
    t["center"] = t["Start"] + (t["End"] - t["Start"]) / 2
    t["transcript_label_start"] = t["center"] - (t["label_size_data_coords"] / 2)
    t["transcript_label_end"] = t["center"] + (t["label_size_data_coords"] / 2)
    t.loc[t["transcript_label_start"] < xmin, "transcript_label_start"] = xmin
    t.loc[t["transcript_label_start"] < xmin, "transcript_label_end"] = t.loc[
        t["transcript_label_start"] < xmin, "label_size_data_coords"
    ]
    t.loc[t["transcript_label_end"] > xmax, "transcript_label_start"] = (
        xmax - t.loc[t["transcript_label_end"] > xmax, "label_size_data_coords"]
    )
    t.loc[t["transcript_label_end"] > xmax, "transcript_label_end"] = xmax
    t["transcript_label_center"] = (
        t["transcript_label_start"]
        + (t["transcript_label_end"] - t["transcript_label_start"]) / 2
    )
    t

    # %%
    transcript_rows = {}
    n_transcripts = transcripts_sorted_by_start_and_length.shape[0]
    transcripts_to_be_placed_current = (
        transcripts_sorted_by_start_and_length.index.to_list()
    )
    transcript_to_be_placed_next = []
    current_row = n_transcripts - 0.5
    current_end = 0
    while transcripts_to_be_placed_current:
        for transcript_id in transcripts_to_be_placed_current:
            transcript_ser = transcripts_sorted_by_start_and_length.loc[transcript_id]
            if transcript_ser[["Start", "transcript_label_start"]].min() < current_end:
                transcript_to_be_placed_next.append(transcript_id)
            else:
                transcript_rows[transcript_id] = current_row
                current_end = transcript_ser[["End", "transcript_label_end"]].max()
        transcripts_to_be_placed_current = transcript_to_be_placed_next
        transcript_to_be_placed_next = []
        current_row -= 1
        current_end = 0
    # make transcript rows start at 0
    transcript_rows = {k: v - current_row - 0.5 for k, v in transcript_rows.items()}
    # %%


    # %%
    ax.clear()
    for (
        transcript_id,
        transcript_ser,
    ) in transcripts_sorted_by_start_and_length.iterrows():
        _plot_transcript(
            transcript_ser=transcript_ser,
            row=transcript_rows[transcript_id],
            ax=ax,
            color='black',
        )
        _plot_transcript_parts(
            df=features.loc[[transcript_id]],
            transcript_ser=transcript_ser,
            row=transcript_rows[transcript_id],
            ax=ax,
            x_axis_size = xmax - xmin,
            color = 'black',
        )

    # somehow the plotting code above resets this

    rectangle_height = 0.3
    ax.set_ylim(- 0.5 + (rectangle_height / 2), max(transcript_rows.values()) + 0.5)

    # display(fig)
    # %%


def get_text_width_data_coordinates(s, ax):
    r = ax.figure.canvas.get_renderer()
    # get window extent in display coordinates
    artist = ax.text(0, 0, s)
    bbox = artist.get_window_extent(renderer=r)
    data_coord_bbox = bbox.transformed(ax.transData.inverted())
    artist.remove()
    # data_coord_bbox.height
    return data_coord_bbox.width


def _plot_transcript(transcript_ser, row, ax, color):
    rectangle_height = 0.3
    ax.hlines(y=row, xmin=transcript_ser.Start, xmax=transcript_ser.End, color=color)
    ax.text(
        x=transcript_ser.loc["transcript_label_center"],
        # this puts it right in the middle - no clear association with a transcript, and may go out of bounds at the bottom row
        # y = row - rectangle_height/2 - (1 - rectangle_height) / 2,
        y=row - rectangle_height / 2 - 0.1,
        s=transcript_ser.gene_name,
        ha="center",
        va="top",
    )

    # # experiment with placing text at the end of transcript like pygenometracks
    # if transcript_ser.Strand == '+':
    #     ax.text(x = transcript_ser.End + x_axis_size * 0.02,
    #             y = row,
    #             s = transcript_ser.gene_name,
    #             ha = 'left',
    #             va = 'center')
    # else:
    #     ax.text(x = transcript_ser.Start -  x_axis_size * 0.02,
    #             y = row,
    #             s = transcript_ser.gene_name,
    #             ha = 'right',
    #             va = 'center')


# %%
def _plot_transcript_parts(df, transcript_ser, row, ax, x_axis_size, color):
    rectangles = []
    rectangle_height = 0.3
    rectangle_height_utrs = 0.15
    for _unused, row_ser in df.iterrows():
        print(row_ser.feature)
        if "UTR" not in row_ser.feature:
            rectangles.append(
                mpatches.Rectangle(
                    xy=(row_ser.Start, row - (rectangle_height / 2)),
                    width=row_ser.End - row_ser.Start,
                    height=rectangle_height,
                    # linewidth=0,
                    color=color,
                )
            )
        else:
            rectangles.append(
                mpatches.Rectangle(
                    xy=(row_ser.Start, row - (rectangle_height_utrs / 2)),
                    width=row_ser.End - row_ser.Start,
                    height=rectangle_height_utrs,
                    # linewidth=0,
                    color=color,
                )
            )
    ax.add_collection(
        matplotlib.collections.PatchCollection(
            rectangles, match_original=True, zorder=3
        ),
    )
    # introns with arrows
    perc_of_axis_between_arrows = 0.03

    arrow_positions = []
    for _unused, ser in (
        pr.PyRanges(transcript_ser.to_frame().T).subtract(pr.PyRanges(df)).df.iterrows()
    ):
        size = ser.End - ser.Start
        n_arrows = np.floor(
            (size) / (perc_of_axis_between_arrows * x_axis_size)
        ).astype(int)
        between_arrows = size / (n_arrows + 1)
        for i in range(1, n_arrows + 1):
            arrow_positions.append(ser.Start + i * between_arrows)
    arrow_positions

    arrow_length_perc_of_x_axis_size = 0.005
    arrow_height = 0.05  # data coordinates size 1 per row
    if transcript_ser.Strand == '+':
        slicer = slice(None, None)
    else:
        slicer = slice(None, None, -1)
    for pos in arrow_positions:
        ax.plot(
            [pos - arrow_length_perc_of_x_axis_size * x_axis_size / 2,
             pos +  arrow_length_perc_of_x_axis_size * x_axis_size / 2],
            [row - arrow_height, row][slicer],
            color = color,
        )
        ax.plot(
            [pos - arrow_length_perc_of_x_axis_size * x_axis_size / 2,
             pos +  arrow_length_perc_of_x_axis_size * x_axis_size / 2],
            [row + arrow_height, row][slicer],
            color = color,
        )
# %%
