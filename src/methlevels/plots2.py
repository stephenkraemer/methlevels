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
    features = (
        df.query('feature in ["exon", "intron", "UTR"]')
        .groupby("transcript_id")
        .apply(lambda df: df.sort_values(["Start", "End"]))
        .droplevel(-1)
    )

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
    ax.get_xlim()
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
    t['center'] = t['Start'] + (t['End'] - t['Start']) / 2
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
    transcript_rows = {k: v - current_row + 0.5 for k, v in transcript_rows.items()}
    # %%

    ax.set_ylim(0, max(transcript_rows.values()) + 0.5)

    for (
        transcript_id,
        transcript_ser,
    ) in transcripts_sorted_by_start_and_length.iterrows():
        _plot_transcript(
            transcript_ser=transcript_ser,
            row=transcript_rows[transcript_id],
            ax=ax,
            x_axis_size=xmax - xmin,
        )
        _plot_transcript_parts(
            df=features.loc[[transcript_id]],
            row=transcript_rows[transcript_id],
            ax=ax,
        )
    # display(fig)


def get_text_width_data_coordinates(s, ax):
    r = ax.figure.canvas.get_renderer()
    # get window extent in display coordinates
    artist = ax.text(0, 0, s)
    bbox = artist.get_window_extent(renderer=r)
    data_coord_bbox = bbox.transformed(ax.transData.inverted())
    artist.remove()
    # data_coord_bbox.height
    return data_coord_bbox.width


def _plot_transcript(transcript_ser, row, ax, x_axis_size):
    rectangle_height = 0.3
    ax.hlines(y=row, xmin=transcript_ser.Start, xmax=transcript_ser.End)
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


def _plot_transcript_parts(df, row, ax):
    rectangles = []
    rectangle_height = 0.3
    for _unused, row_ser in df.iterrows():
        rectangles.append(
            mpatches.Rectangle(
                xy=(row_ser.Start, row - (rectangle_height / 2)),
                width=row_ser.End - row_ser.Start,
                height=rectangle_height,
                # linewidth=0,
                color="blue",
            )
        )
    ax.add_collection(
        matplotlib.collections.PatchCollection(
            rectangles, match_original=True, zorder=3
        ),
    )
