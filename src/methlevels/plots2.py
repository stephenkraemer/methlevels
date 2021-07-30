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
import pyranges as pr

from methlevels import MethStats
from methlevels.utils import NamedColumnsSlice as ncls

import matplotlib.patches as mpatches
import matplotlib.collections
from typing import Literal


def plot_gene_model(
    df,
    ax,
    rectangle_height=0.5,
    rectangle_height_utrs=0.25,
    perc_of_axis_between_arrows=0.03,
    arrow_length_perc_of_x_axis_size=0.01,
    arrow_height=0.15,
    gene_label_size=5,
    bp_scale: Literal["bp", "Kb", "Mb"] = "bp",
    format_str = '{:.2f}',
):
    """Plot gene model



    Parameters
    ----------
    df
       required columns: 'Chromosome', 'feature', 'Start', 'End', 'Strand', 'transcript_id', 'gene_name'
       other columns are ignored
        only rows representing transcript, exon and utr features will be used.
        other rows (eg gene features) are allowed, but will be ignored
    rectangle_height
        height of exon rectangles, given as fraction of the distance between two transcript lines
    rectangle_height_utrs
        height of utr rectangles, given as fraction of the distance between two transcript lines
    arrow_length_perc_of_x_axis_size
        given as fraction of the x axis length, ie it gives the length in axes coordinates
    arrow_height
        given as fraction of the distance between two transcript lines
    perc_of_axis_between_arrows
        minimal distance between two arrows on a transcript
        algorithm will decide for each intron individually how many arrows can be placed while
        respecting this distance
        given as fraction of the x axis length, ie it gives the length in axes coordinates
    """

    assert all(
        [
            x in df.columns
            for x in [
                "Chromosome",
                "feature",
                "Start",
                "End",
                "Strand",
                "transcript_id",
                "gene_name",
            ]
        ]
    )

    transcripts_sorted_by_start_and_length = _get_sorted_transcripts(df)
    sorted_transcript_parts = _get_sorted_transcript_parts(df)

    # Setting axis labels MUST be done before determining label sizes in next step (?)
    xmin = transcripts_sorted_by_start_and_length.Start.min()
    xmax = transcripts_sorted_by_start_and_length.End.max()
    ax.set_xlim(xmin, xmax)

    transcripts_sorted_by_start_and_length_with_label_pos = (
        _add_transcript_label_position_columns(
            transcripts_sorted_by_start_and_length, ax, xmin, xmax
        )
    )

    transcript_rows = _compute_transcript_row_placement(
        transcripts_sorted_by_start_and_length_with_label_pos
    )

    _add_transcript_transcript_parts_and_arrows(
        transcripts_sorted_by_start_and_length,
        transcript_rows,
        sorted_transcript_parts,
        ax,
        xmin,
        xmax,
        rectangle_height,
        rectangle_height_utrs,
        perc_of_axis_between_arrows,
        arrow_length_perc_of_x_axis_size,
        arrow_height,
        gene_label_size,
        bp_scale,
        format_str,
    )


def get_text_width_data_coordinates(s, ax):
    r = ax.figure.canvas.get_renderer()
    # get window extent in display coordinates
    artist = ax.text(0, 0, s)
    bbox = artist.get_window_extent(renderer=r)
    data_coord_bbox = bbox.transformed(ax.transData.inverted())
    artist.remove()
    # data_coord_bbox.height
    return data_coord_bbox.width


def _plot_transcript(transcript_ser, row, ax, color, rectangle_height, gene_label_size):
    ax.hlines(y=row, xmin=transcript_ser.Start, xmax=transcript_ser.End, color=color)
    ax.text(
        x=transcript_ser.loc["transcript_label_center"],
        # this puts it right in the middle - no clear association with a transcript, and may go out of bounds at the bottom row
        # y = row - rectangle_height/2 - (1 - rectangle_height) / 2,
        y=row - rectangle_height / 2 - 0.1,
        s=transcript_ser.gene_name,
        ha="center",
        va="top",
        size=gene_label_size,
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
def _plot_transcript_parts(
    df,
    transcript_ser,
    row,
    ax,
    x_axis_size,
    color,
    rectangle_height,
    rectangle_height_utrs,
    perc_of_axis_between_arrows,
    arrow_length_perc_of_x_axis_size,
    arrow_height,
):
    rectangles = []
    for _unused, row_ser in df.iterrows():
        print(row_ser.feature)
        if "UTR" not in row_ser.feature:
            rectangles.append(
                mpatches.Rectangle(  # type: ignore
                    xy=(row_ser.Start, row - (rectangle_height / 2)),
                    width=row_ser.End - row_ser.Start,
                    height=rectangle_height,
                    # linewidth=0,
                    color=color,
                )
            )
        else:
            rectangles.append(
                mpatches.Rectangle(  # type: ignore
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

    arrow_positions = []
    for _unused, ser in (  # type: ignore
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

    if transcript_ser.Strand == "+":
        slicer = slice(None, None)
    else:
        slicer = slice(None, None, -1)
    for pos in arrow_positions:
        ax.plot(
            [
                pos - arrow_length_perc_of_x_axis_size * x_axis_size / 2,
                pos + arrow_length_perc_of_x_axis_size * x_axis_size / 2,
            ],
            [row - arrow_height, row][slicer],
            color=color,
        )
        ax.plot(
            [
                pos - arrow_length_perc_of_x_axis_size * x_axis_size / 2,
                pos + arrow_length_perc_of_x_axis_size * x_axis_size / 2,
            ],
            [row + arrow_height, row][slicer],
            color=color,
        )


# %%


def _get_sorted_transcripts(df):
    # sort start ascending, length descending (by sorting negative length)
    transcripts_sorted_by_start_and_length = (
        df.query('feature == "transcript"')
        .assign(length_neg=lambda df: -(df.End - df.Start))
        .sort_values(["Start", "length_neg"])
        .set_index("transcript_id")
    )
    return transcripts_sorted_by_start_and_length


def _get_sorted_transcript_parts(df):
    transcript_parts_dfs = []
    for _unused, group_df in df.groupby("transcript_id"):  # type: ignore
        gr_exons = pr.PyRanges(group_df.query('feature == "exon"'))
        gr_utrs = pr.PyRanges(group_df.query('"UTR" in feature'))
        gr_exons = gr_exons.subtract(gr_utrs)
        transcript_parts_dfs.append(pd.concat([gr_exons.df, gr_utrs.df], axis=0))

    # transcript parts (exons, utrs) sorted by ascending Start per transcript
    transcript_parts_sorted = (
        pd.concat(transcript_parts_dfs, axis=0)
        .groupby("transcript_id")
        .apply(lambda df: df.sort_values(["Start", "End"]))
        .droplevel(-1)
    )

    n_utr_and_exons_features_expected = (
        df.query('feature == "exon" or "UTR" in feature')
        .drop_duplicates(subset=["Chromosome", "Start", "End", "transcript_id"])
        .shape[0]
    )
    assert n_utr_and_exons_features_expected == transcript_parts_sorted.shape[0]

    return transcript_parts_sorted


def _add_transcript_label_position_columns(
    transcripts_sorted_by_start_and_length, ax, xmin, xmax
):
    """

    Setting axis labels MUST be done before determining label sizes in next step (?)
    """

    # Setting axis labels MUST be done before determining label sizes in next step (?)
    assert ax.get_xlim() == (xmin, xmax)

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
    return t


def _compute_transcript_row_placement(transcripts_sorted_by_start_and_length):

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
    return transcript_rows


def _add_transcript_transcript_parts_and_arrows(
    transcripts_sorted_by_start_and_length,
    transcript_rows,
    sorted_transcript_parts,
    ax,
    xmin,
    xmax,
    rectangle_height,
    rectangle_height_utrs,
    perc_of_axis_between_arrows,
    arrow_length_perc_of_x_axis_size,
    arrow_height,
    gene_label_size,
    bp_scale: Literal["bp", "kb", "mb"],
        format_str,
):

    assert ax.get_xlim() == (xmin, xmax)

    for (
        transcript_id,
        transcript_ser,
    ) in transcripts_sorted_by_start_and_length.iterrows():
        _plot_transcript(
            transcript_ser=transcript_ser,
            row=transcript_rows[transcript_id],
            ax=ax,
            color="black",
            rectangle_height=rectangle_height,
            gene_label_size=gene_label_size,
        )
        _plot_transcript_parts(
            df=sorted_transcript_parts.loc[[transcript_id]],
            transcript_ser=transcript_ser,
            row=transcript_rows[transcript_id],
            ax=ax,
            x_axis_size=xmax - xmin,
            color="black",
            rectangle_height=rectangle_height,
            rectangle_height_utrs=rectangle_height_utrs,
            perc_of_axis_between_arrows=perc_of_axis_between_arrows,
            arrow_length_perc_of_x_axis_size=arrow_length_perc_of_x_axis_size,
            arrow_height=arrow_height,
        )

    # somehow the plotting code above resets this
    ax.set_ylim(-0.5 + (rectangle_height / 2), max(transcript_rows.values()) + 0.5)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(_bp_ticklabel_format_func(bp_scale, format_str))
    )

    ax.set(xlabel=f"Position ({bp_scale})")

    # despine
    ax.tick_params(
        axis="y",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
        labelsize=0,
        length=0,
    )
    ax.spines["left"].set_visible(False)


def _bp_ticklabel_format_func(bp_scale, format_str):
    def wrapped(value, tick_number):
        return format_str.format(
            value / {"bp": 1, "Kb": 1000, "Mb": 1_000_000}[bp_scale]
        )
    return wrapped
