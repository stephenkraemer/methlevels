from typing import Optional, Union, Dict, Tuple
import codaplot.utils as coutils

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
    df: pd.DataFrame,
    ax: Axes,
    xlabel: str,
    order_of_magnitude: Optional[int] = None,
    offset: Optional[Union[float, bool]] = None,
    roi: Optional[Tuple[float, float]] = None,
    rectangle_height=0.5,
    rectangle_height_utrs=0.25,
    perc_of_axis_between_arrows=0.03,
    arrow_length_perc_of_x_axis_size=0.01,
    arrow_height=0.15,
    gene_label_size=5,
):
    """Plot gene model



    Parameters
    ----------
    df
       required columns: 'Chromosome', 'Feature', 'Start', 'End', 'Strand', 'transcript_id', 'gene_name'
       other columns are ignored
        only rows representing transcript, exon and utr features will be used.
        other rows (eg gene features) are allowed, but will be ignored
    order_of_magniture
        if specified, forces scientific notations with this oom. you can only specify oom or offset
        **NOTE** this feature may be untested
    offset
        if specified, forces offest notation. you can only specify oom or offset
    roi
        if None, axes limits are set to the minimum and maximum genomic position in df
        if set, can be used to zoom in. this is often useful, because the df will usually contain full feature intervals, but we may only be interest in the part of the intervals which overlaps with an roi
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

    if (offset is not None) and not offset:
        offset = None

    assert not ((order_of_magnitude is not None) and (offset is not None))

    # we will set the axis limits to roi later, if not None
    # without a priori restricting the df to the roi,
    # constrained layout fails, haven't checked it further yet
    if roi:
        df = (  # type: ignore
            pr.PyRanges(df)
            .intersect(
                pr.PyRanges(
                    chromosomes=[df.Chromosome.iloc[0]], starts=[roi[0]], ends=[roi[1]]
                )
            )
            .df
        )

    assert all(
        [
            x in df.columns
            for x in [
                "Chromosome",
                "Feature",
                "Start",
                "End",
                "Strand",
                "transcript_id",
                "gene_name",
            ]
        ]
    )

    print("Working with str")
    df["Chromosome"] = df["Chromosome"].astype(str)
    # bug in groupby-apply when df['Chromosome'].dtype == "category"
    # generally, there where multiple categorical related bugs in the past, and we don't
    # need a categorical here, so lets completely avoid it by working with strings

    transcripts_sorted_by_start_and_length = _get_sorted_transcripts(df)
    sorted_transcript_parts = _get_sorted_transcript_parts(df)

    # Setting axis limits may have to be done before determining label sizes in next step (?)
    if roi:
        xmin, xmax = roi
    else:
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
        transcripts_sorted_by_start_and_length=transcripts_sorted_by_start_and_length,
        transcript_rows=transcript_rows,
        sorted_transcript_parts=sorted_transcript_parts,
        ax=ax,
        xlabel=xlabel,
        xmin=xmin,
        xmax=xmax,
        rectangle_height=rectangle_height,
        rectangle_height_utrs=rectangle_height_utrs,
        perc_of_axis_between_arrows=perc_of_axis_between_arrows,
        arrow_length_perc_of_x_axis_size=arrow_length_perc_of_x_axis_size,
        arrow_height=arrow_height,
        gene_label_size=gene_label_size,
    )

    _format_axis_with_offset_or_order_of_magnitude(ax, offset, order_of_magnitude)


def _format_axis_with_offset_or_order_of_magnitude(ax, offset, order_of_magnitude):
    # currently bug - does not remove trailing zeros from offset
    # ax.xaxis.set_major_formatter(mticker.ScalarFormatterQuickfixed(useOffset=True))
    ax.xaxis.set_major_formatter(coutils.ScalarFormatterQuickfixed(useOffset=True))
    if offset and isinstance(offset, bool):
        offset = coutils.find_offset(ax=ax)
    if offset:
        ax.ticklabel_format(axis="x", useOffset=offset)  # type: ignore
    if order_of_magnitude:
        ax.ticklabel_format(
            axis="x", scilimits=(order_of_magnitude, order_of_magnitude)
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
        print(row_ser.Feature)
        if "UTR" not in row_ser.Feature:
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
        df.query('Feature == "transcript"')
        .assign(length_neg=lambda df: -(df.End - df.Start))
        .sort_values(["Start", "length_neg"])
        .set_index("transcript_id")
    )
    return transcripts_sorted_by_start_and_length


def _get_sorted_transcript_parts(df):

    transcript_parts_dfs = []
    for _unused, group_df in df.groupby("transcript_id"):  # type: ignore
        gr_utrs = pr.PyRanges(group_df.query('"UTR" in Feature'))
        if not gr_utrs.empty:
            gr_utrs_df = gr_utrs.df.assign(
                Chromosome=lambda df: df["Chromosome"].astype(str),
                Strand=lambda df: df["Strand"].astype(str),
            )
        else:
            gr_utrs_df = pd.DataFrame()
        gr_exons = pr.PyRanges(group_df.query('Feature == "exon"'))
        if not gr_exons.empty:
            if not gr_utrs.empty:
                gr_exons = gr_exons.subtract(gr_utrs)
            if not gr_exons.empty:
                gr_exons_df = gr_exons.df.assign(
                    Chromosome=lambda df: df["Chromosome"].astype(str),
                    Strand=lambda df: df["Strand"].astype(str),
                )
            else:
                gr_exons_df = pd.DataFrame()
        else:
            gr_exons_df = pd.DataFrame()
        transcript_parts_dfs.append(pd.concat([gr_exons_df, gr_utrs_df], axis=0))

    # transcript parts (exons, utrs) sorted by ascending Start per transcript
    transcript_parts_sorted = (
        pd.concat(transcript_parts_dfs, axis=0)
        # [['Chromosome', 'Start', 'End', 'transcript_id']]
        # .assign(Chromosome = lambda df: df.Chromosome.astype(str))
        .groupby("transcript_id")
        .apply(lambda df: df.sort_values(["Start", "End"]))
        .droplevel(-1)
    )

    n_utr_and_exons_features_expected = (
        df.query('Feature == "exon" or "UTR" in Feature')
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
    xlabel,
    xmin,
    xmax,
    rectangle_height,
    rectangle_height_utrs,
    perc_of_axis_between_arrows,
    arrow_length_perc_of_x_axis_size,
    arrow_height,
    gene_label_size,
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

    ax.set(xlabel=xlabel)

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


def plot_genomic_region_track(
    granges_gr,
    ax,
    order_of_magnitude: Optional[int] = None,
    offset: Optional[Union[float, bool]] = None,
    no_coords=False,
    roi: Optional[Tuple[int, int]] = None,
    color: Optional[Union[str, Tuple]] = None,
    palette: Optional[Dict[str, str]] = None,
    show_names=False,
    ax_abs_height: Optional[float] = None,
    label_size=6,
    ymargin=0.05,
):
    """Plot a single, non-overlapping set of genomic regions onto an Axes

    optionally, place names below some of the regions

    Parameters
    ----------
    granges_gr
        columns: if show_names: name. name column may contain empty rows, indicated by '' or np.nan or None
        this is expected to be restricted to the ROI to be plottet
        the start/end args allow setting xlim to zoom in, but the chromosome must already be unique in this input dataframe
    ax
    order_of_magniture
        if specified, forces scientific notations with this oom. you can only specify oom or offset
        **NOTE** this feature may be untested
    offset
        if specified, forces offest notation. you can only specify oom or offset
    no_coords
        if True, the x axis is not shown (default). the y axis is never shown.
    ax_abs_size
        size of Axes in inch, to allow fitting the region labels into the plot
        required if show_names = True
    roi: optionally, zoom into the data by using xlim = roi
    color, palette
        optional either i) color = specify single color for all regions ii) palette = optional dict mapping region name -> hex color
    show_names
        show text label below each region with a name in the name column. regions without name value do not get a label without error.
    label_size
        size of the region name label (for show_names = True, otherwise ignored)
    ymargin
        margin between plot elements and axis as percentage of axes (aesthetic parameter which may be interesting to adjust the look at different plot sizes)

    """

    if (offset is not None) and not offset:
        offset = None
    assert not ((order_of_magnitude is not None) and (offset is not None))

    label_padding_in = 0.1 / 2.54

    if color:
        assert palette is None
    if palette:
        assert color is None
        assert (
            np.array(list(sorted(palette.keys())))
            == np.sort(granges_gr.df["name"].unique())
        ).all()
    if show_names:
        assert ax_abs_height is not None

    if show_names:
        label_height_in = label_size * 1 / 72
        ax_fraction_for_rectangles = (
            1 - ymargin - (label_height_in + label_padding_in) / ax_abs_height
        )
    else:
        ax_fraction_for_rectangles = 1 - ymargin

    if roi:
        granges_df = granges_gr[roi[0] : roi[1]].df
    else:
        granges_df = granges_gr.df
        roi = granges_df["Start"].min(), granges_df["End"].max()
    ax.set_xlim(roi)

    ax.set(ylim=(0, 1))

    if no_coords:
        coutils.strip_all_axis(ax)
    else:
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            labelleft=False,
            labelsize=0,
            length=0,
        )
        ax.spines["left"].set_visible(False)

    rectangles = []
    for _unused, row_ser in granges_df.iterrows():  # type: ignore
        if palette:
            color = palette[row_ser["name"]]
        rectangles.append(
            mpatches.Rectangle(  # type: ignore
                xy=(row_ser.Start, 1 - ax_fraction_for_rectangles),
                width=row_ser.End - row_ser.Start,
                height=ax_fraction_for_rectangles,
                linewidth=0,
                color=color,
            )
        )
        if show_names:
            if row_ser["name"]:
                # place text in middle of displayed region, not in middle of full interval, parts of which may lie outside of displayed region
                ax.text(
                    x=min(row_ser["End"], roi[1])
                    - (
                        (min(row_ser["End"], roi[1]) - max(row_ser["Start"], roi[0]))
                        / 2
                    ),
                    y=ymargin,
                    s=row_ser["name"],
                    ha="center",
                    va="bottom",
                    size=label_size,
                )
    ax.add_collection(
        matplotlib.collections.PatchCollection(
            rectangles, match_original=True, zorder=3
        ),
    )

    _format_axis_with_offset_or_order_of_magnitude(ax, offset, order_of_magnitude)
