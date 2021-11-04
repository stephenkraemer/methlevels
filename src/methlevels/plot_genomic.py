from typing import Optional, Union, Dict, Tuple
import codaplot.utils as coutils
from adjustText import adjust_text

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

print("reloaded plot genomic")


def plot_gene_model_get_height(
    df: pd.DataFrame,
    axeswidth,
    xlim,
    exon_rect_height_in=0.3 / 2.54,
    space_between_transcripts_in=0.3 / 2.54,
    space_between_transcript_label_and_transcript_box_in=0.1 / 2.54,
    label_fontsize=6,
):

    # improve: the logic is copy pasted from plot_gene_model and needs to be adapted if the plotting function changes

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

    # we will set the axis limits to roi later, if not None
    # without a priori restricting the df to the roi,
    # constrained layout fails, haven't checked it further yet
    if xlim:
        df = (  # type: ignore
            pr.PyRanges(df)
            .intersect(
                pr.PyRanges(
                    chromosomes=[df.Chromosome.iloc[0]],
                    starts=[xlim[0]],
                    ends=[xlim[1]],
                )
            )
            .df
        )

    # bug in groupby-apply when df['Chromosome'].dtype == "category"
    # generally, there where multiple categorical related bugs in the past, and we don't
    # need a categorical here, so lets completely avoid it by working with strings
    df["Chromosome"] = df["Chromosome"].astype(str)

    transcripts_sorted_by_start_and_length = _get_sorted_transcripts(df)

    # set xlim prior to getting text width in data coords
    if xlim:
        xmin, xmax = xlim
    else:
        xmin = transcripts_sorted_by_start_and_length.Start.min()
        xmax = transcripts_sorted_by_start_and_length.End.max()

    fig, ax = plt.subplots(1, 1, dpi=180, figsize=(axeswidth, 4))
    ax.set_xlim(xmin, xmax)

    transcripts_sorted_by_start_and_length_with_label_pos = (
        _add_transcript_label_position_columns(
            transcripts_sorted_by_start_and_length, ax, xmin, xmax
        )
    )

    # transcript rows for historical reasons 0.5, 1.5..
    transcript_rows = _compute_transcript_row_placement(
        transcripts_sorted_by_start_and_length_with_label_pos
    )
    n_rows = max(transcript_rows.values()) + 0.5

    # bottom margin so that text does not align with y axis spline
    # exon rects and arrows were slightly cut at the top; did not investigate
    # further yet, just also added margin at the top
    ymargin_bottom_and_top_spacer_in = 0.1 / 2.54

    max_text_width_in, max_text_height_in = _get_max_text_height_width_in_x(
        labels=transcripts_sorted_by_start_and_length_with_label_pos[
            "gene_name"
        ].unique(),
        fontsize=label_fontsize,
        ax=ax,
        x="inch",
    )

    size_of_one_transcript_box_with_spacer = (
        max_text_height_in
        + space_between_transcript_label_and_transcript_box_in
        + exon_rect_height_in
        + space_between_transcripts_in
    )

    axes_height_in = (
        ymargin_bottom_and_top_spacer_in
        + size_of_one_transcript_box_with_spacer * n_rows
        - space_between_transcripts_in
        + ymargin_bottom_and_top_spacer_in
    )

    return axes_height_in


def plot_gene_model(
    df: pd.DataFrame,
    ax: Axes,
    xlabel: str,
    order_of_magnitude: Optional[int] = None,
    offset: Optional[Union[float, bool]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    exon_rect_height_in=0.3 / 2.54,
    rectangle_height_utrs_frac=0.5,
    space_between_transcripts_in=0.3 / 2.54,
    space_between_transcript_label_and_transcript_box_in=0.1 / 2.54,
    space_between_arrows_in=1 / 2.54,
    arrow_length_in=0.075 / 2.54,
    arrow_height_frac=0.5,
    label_fontsize=6,
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
    xlim
        if None, axes limits are set to the minimum and maximum genomic position in df
        if set, can be used to zoom in. this is often useful, because the df will usually contain full feature intervals, but we may only be interest in the part of the intervals which overlaps with an xlim
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
    y_bottom_margin
        preliminary solution to avoid problems with feature labels overlapping the x axis (?)
        transcripts are centered around y=0.5, 1.5, ...
        ylim max is set to max row, e.g. 1.5 + rectangle_height / 2
        ylim min is set to y_bottom_margin, e.g. -0.5, to give some margin at the bottom of the plot. adjust depending on plot size. this should be better automated in the future
    """

    # argument handling
    if (offset is not None) and not offset:
        offset = None
    assert not ((order_of_magnitude is not None) and (offset is not None))
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

    # we will set the axis limits to roi later, if not None
    # without a priori restricting the df to the roi,
    # constrained layout fails, haven't checked it further yet
    if xlim:
        df = (  # type: ignore
            pr.PyRanges(df)
            .intersect(
                pr.PyRanges(
                    chromosomes=[df.Chromosome.iloc[0]],
                    starts=[xlim[0]],
                    ends=[xlim[1]],
                )
            )
            .df
        )

    # bug in groupby-apply when df['Chromosome'].dtype == "category"
    # generally, there where multiple categorical related bugs in the past, and we don't
    # need a categorical here, so lets completely avoid it by working with strings
    df["Chromosome"] = df["Chromosome"].astype(str)

    transcripts_sorted_by_start_and_length = _get_sorted_transcripts(df)
    sorted_transcript_parts = _get_sorted_transcript_parts(df)

    # set xlim prior to getting text width in data coords
    if xlim:
        xmin, xmax = xlim
    else:
        xmin = transcripts_sorted_by_start_and_length.Start.min()
        xmax = transcripts_sorted_by_start_and_length.End.max()
    ax.set_xlim(xmin, xmax)

    # we set ylim to (0, 1) and then use absolute size -> data coord mapping to place the artists
    ax.set_ylim(0, 1)

    transcripts_sorted_by_start_and_length_with_label_pos = (
        _add_transcript_label_position_columns(
            transcripts_sorted_by_start_and_length, ax, xmin, xmax
        )
    )

    # transcript rows for historical reasons 0.5, 1.5..
    transcript_rows = _compute_transcript_row_placement(
        transcripts_sorted_by_start_and_length_with_label_pos
    )
    n_rows = max(transcript_rows.values()) + 0.5

    # bottom margin so that text does not align with y axis spline
    # exon rects and arrows were slightly cut at the top; did not investigate
    # further yet, just also added margin at the top
    ymargin_bottom_and_top_spacer_in = 0.1 / 2.54

    max_text_width_in, max_text_height_in = _get_max_text_height_width_in_x(
        labels=transcripts_sorted_by_start_and_length_with_label_pos[
            "gene_name"
        ].unique(),
        fontsize=label_fontsize,
        ax=ax,
        x="inch",
    )
    (
        max_text_width_data_coords,
        max_text_height_data_coords,
    ) = _get_max_text_height_width_in_x(
        labels=transcripts_sorted_by_start_and_length_with_label_pos[
            "gene_name"
        ].unique(),
        fontsize=label_fontsize,
        ax=ax,
        x="data_coordinates",
    )

    size_of_one_transcript_row_with_spacer = (
        max_text_height_in
        + space_between_transcript_label_and_transcript_box_in
        + exon_rect_height_in
        + space_between_transcripts_in
    )

    # improve: this is not used here, a copy of this snippet is used in get_plot_gene_model_height
    axes_height_in = (
        ymargin_bottom_and_top_spacer_in
        + size_of_one_transcript_row_with_spacer * n_rows
        - space_between_transcripts_in
        + ymargin_bottom_and_top_spacer_in
    )

    # map to historical size params
    # =============================

    # fig, ax = plt.subplots(1, 1, dpi=180, figsize=(cm(8), axes_height_in))
    # fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    # ax.set_ylim(0, 1)
    # ax.set_xlim(xmin, xmax)

    # - transcript_rows should give center of transcript line in data coords
    transcript_rows_2 = {}
    for transcript_id, row_idx in transcript_rows.items():
        transcript_rows_2[transcript_id] = coutils.convert_inch_to_data_coords(
            size=(
                ymargin_bottom_and_top_spacer_in
                + size_of_one_transcript_row_with_spacer * (row_idx + 0.5)
                - space_between_transcripts_in
                - exon_rect_height_in / 2
            ),
            ax=ax,
        )[1]

    exon_height_data_coords = coutils.convert_inch_to_data_coords(
        size=exon_rect_height_in, ax=ax
    )[1]
    utr_height_data_coords = exon_height_data_coords * rectangle_height_utrs_frac
    perc_of_xaxis_between_arrows = coutils.convert_inch_to_data_coords(
        size=space_between_arrows_in, ax=ax
    )[0]
    arrow_length_data_coords = coutils.convert_inch_to_data_coords(
        size=arrow_length_in, ax=ax
    )[0]
    arrow_height_data_coords = arrow_height_frac * exon_height_data_coords
    exon_text_spacer_height_data_coords = coutils.convert_inch_to_data_coords(
        size=space_between_transcript_label_and_transcript_box_in, ax=ax
    )[1]

    _add_transcript_transcript_parts_and_arrows(
        transcripts_sorted_by_start_and_length=transcripts_sorted_by_start_and_length,
        transcript_rows=transcript_rows_2,
        sorted_transcript_parts=sorted_transcript_parts,
        ax=ax,
        xlabel=xlabel,
        xmin=xmin,
        xmax=xmax,
        exon_height_data_coords=exon_height_data_coords,
        max_text_height_data_coords=max_text_height_data_coords,
        exon_text_spacer_height_data_coords=exon_text_spacer_height_data_coords,
        utr_height_data_coords=utr_height_data_coords,
        dist_between_arrows_data_coords=perc_of_xaxis_between_arrows,
        arrow_length_data_coords=arrow_length_data_coords,
        arrow_height_data_coords=arrow_height_data_coords,
        gene_label_size=label_fontsize,
    )

    _format_axis_with_offset_or_order_of_magnitude(ax, offset, order_of_magnitude)

    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set(xlabel=xlabel)


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


def _plot_transcript(
    transcript_ser,
    row,
    ax,
    color,
    exon_height_data_coords,
    max_text_height_data_coords,
    exon_text_spacer_height_data_coords,
    gene_label_size,
):

    ax.hlines(y=row, xmin=transcript_ser.Start, xmax=transcript_ser.End, color=color)

    # improve:
    # some names may have less height than others, eg. may be completely lowercase while others are uppercase
    # some names may have 'pg' characters, some may not
    # currently: align everything at baseline
    ax.text(
        x=transcript_ser.loc["transcript_label_center"],
        # this puts it right in the middle - no clear association with a transcript, and may go out of bounds at the bottom row
        # y = row - rectangle_height/2 - (1 - rectangle_height) / 2,
        y=row
        - exon_height_data_coords / 2
        - exon_text_spacer_height_data_coords
        - max_text_height_data_coords,
        s=transcript_ser.gene_name,
        ha="center",
        va="baseline",
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
    exon_height_data_coords,
    rectangle_height_utrs,
    dist_between_arrows_data_coords,
    arrow_length_data_coords,
    arrow_height_data_coords,
):
    """

    Parameters
    ----------
    df
        info about utrs and exons, may be empty
    """

    # if df (of exons and utrs) is not empty,
    # Draw exon and UTR rectangles patches
    rectangles = []
    for _unused, row_ser in df.iterrows():
        if "UTR" not in row_ser.Feature:
            rectangles.append(
                mpatches.Rectangle(  # type: ignore
                    xy=(row_ser.Start, row - (exon_height_data_coords / 2)),
                    width=row_ser.End - row_ser.Start,
                    height=exon_height_data_coords,
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
    if rectangles:
        ax.add_collection(
            matplotlib.collections.PatchCollection(
                rectangles, match_original=True, zorder=3
            ),
        )

    # decorate introns with arrows
    arrow_positions = []
    intron_intervals_df = (
        pr.PyRanges(transcript_ser.to_frame().T).subtract(pr.PyRanges(df)).df
    )
    for _unused, ser in intron_intervals_df.iterrows():
        # arrows are plotted with tip at the arrow positions
        # -> first arrow at exon/utr + dist_between_arrows_data_coords + arrow_length_data_coords
        arrow_positions = np.arange(
            ser["Start"] + dist_between_arrows_data_coords + arrow_length_data_coords,
            ser["End"] - dist_between_arrows_data_coords * 0.5,
            dist_between_arrows_data_coords + arrow_length_data_coords,
        )
    arrow_positions

    if transcript_ser.Strand == "+":
        slicer = slice(None, None)
    else:
        slicer = slice(None, None, -1)
    for pos in arrow_positions:
        ax.plot(
            [
                pos - arrow_length_data_coords,
                pos,
            ],
            [row - arrow_height_data_coords, row][slicer],
            color=color,
        )
        ax.plot(
            [
                pos - arrow_length_data_coords,
                pos,
            ],
            [row + arrow_height_data_coords, row][slicer],
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

    transcript_parts_sorted = pd.concat(transcript_parts_dfs, axis=0)
    if not transcript_parts_sorted.empty:
        transcript_parts_sorted = (
            transcript_parts_sorted
            # transcript parts (exons, utrs) sorted by ascending Start per transcript
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
    """Place transcript labels centered on transcript line if possible

    If transcript labels go below xmin or above xmax, shift them inwards

    Setting axis labels must be done before determining label sizes in next step
    """

    # Setting axis labels MUST be done before determining label sizes in next step (?)
    assert ax.get_xlim() == (xmin, xmax)

    t = transcripts_sorted_by_start_and_length

    t["label_size_data_coords"] = t.gene_name.apply(
        get_text_width_data_coordinates, ax=ax
    )
    # transcript label will be placed at the center
    # of the transcript line initially
    t["center"] = t["Start"] + (t["End"] - t["Start"]) / 2

    # Calculate label starts and ends if placed at the center of the transcript line
    t["transcript_label_start"] = t["center"] - (t["label_size_data_coords"] / 2)
    t["transcript_label_end"] = t["center"] + (t["label_size_data_coords"] / 2)

    # transcript labels which start below xmin are shifted such that they start at xmin; they will no longer be centered with respect to the transcript line
    t.loc[t["transcript_label_start"] < xmin, "transcript_label_start"] = xmin
    t.loc[t["transcript_label_start"] < xmin, "transcript_label_end"] = t.loc[
        t["transcript_label_start"] < xmin, "label_size_data_coords"
    ]

    # transcript labels which end beyond xmax are shifted left such that they end at xmax
    # not that this could in theory lead to the label start lying below xmin
    t.loc[t["transcript_label_end"] > xmax, "transcript_label_start"] = (
        xmax - t.loc[t["transcript_label_end"] > xmax, "label_size_data_coords"]
    )
    t.loc[t["transcript_label_end"] > xmax, "transcript_label_end"] = xmax

    # calculate the center of the transcript label
    # (which is not necessarily the center of the transcript line)
    t["transcript_label_center"] = (
        t["transcript_label_start"]
        + (t["transcript_label_end"] - t["transcript_label_start"]) / 2
    )
    return t


def _compute_transcript_row_placement(transcripts_sorted_by_start_and_length):
    """Places transcripts in multiple rows to avoid overlaps

    Transcripts are placed based on transcript_label_start and transcript_label_end, which indicate the outmost position of the transcript line or the transcript label. So both transcript label and line overlaps are avoided.

    Longer transcripts are placed first, smaller gaps are filled with shorter transcripts if the next transcript in line cannot be placed.

    Returns
    -------
    dictionary mapping transcript_ids to row ids
        row_ids are 0.5, 1.5, etc.
    """

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
    exon_height_data_coords,
    max_text_height_data_coords,
    utr_height_data_coords,
    dist_between_arrows_data_coords,
    arrow_length_data_coords,
    arrow_height_data_coords,
    exon_text_spacer_height_data_coords,
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
            exon_height_data_coords=exon_height_data_coords,
            max_text_height_data_coords=max_text_height_data_coords,
            exon_text_spacer_height_data_coords=exon_text_spacer_height_data_coords,
            gene_label_size=gene_label_size,
        )
        if transcript_id in sorted_transcript_parts.index:
            curr_transcript_parts = sorted_transcript_parts.loc[[transcript_id]]
        else:
            # sorted_transcript_parts could be completely empty
            # or undefined for the current transcript id
            # in which case sorted_transcript_parts holds an empty dataframe
            # without columns index
            curr_transcript_parts = pd.DataFrame(columns=["Chromosome", "Start", "End"])
        _plot_transcript_parts(
            df=curr_transcript_parts,
            transcript_ser=transcript_ser,
            row=transcript_rows[transcript_id],
            ax=ax,
            x_axis_size=xmax - xmin,
            color="black",
            exon_height_data_coords=exon_height_data_coords,
            rectangle_height_utrs=utr_height_data_coords,
            dist_between_arrows_data_coords=dist_between_arrows_data_coords,
            arrow_length_data_coords=arrow_length_data_coords,
            arrow_height_data_coords=arrow_height_data_coords,
        )


def plot_genomic_region_track(
    granges_gr,
    ax,
    order_of_magnitude: Optional[int] = None,
    offset: Optional[Union[float, bool]] = None,
    xlim: Optional[Tuple[int, int]] = None,
    color: Union[str, Tuple] = "gray",
    palette: Optional[Dict[str, str]] = None,
    show_names=False,
    title: Optional[str] = None,
    title_size: Optional[float] = None,
    # size aesthetics
    space_between_label_and_patch_in=0.1 / 2.54,
    space_between_rows_in=0.1 / 2.54,
    patch_height_in=0.4 / 2.54,
    label_fontsize: Optional[float] = None,
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
    xlim
        optionally, zoom into the data by using xlim
    color
        default color
    palette
        optional dict mapping region name -> hex color, may also only contain some of the region names, then fallback is color
    show_names
        show text label below each region with a name in the name column. regions without name value do not get a label without error.
    label_size
        size of the region name label (for show_names = True, otherwise ignored)
        if none defaults to mpl.rcParams['xtick.labelsize']
    title
        axes title, added at title_side; label size is taken from mpl.rcParams["axes.titlesize"]
    title_size
        fontsize for track title, if None defaults to mpl.rcParams["axes.titlesize"]
    """

    # arg handling
    if label_fontsize is None:
        label_fontsize = mpl.rcParams["xtick.labelsize"]
    if not title_size:
        title_size = mpl.rcParams["axes.titlesize"]
    if (offset is not None) and not offset:
        offset = None
    assert not ((order_of_magnitude is not None) and (offset is not None))
    # granges must have unique index for operations in this function
    assert not granges_gr.df.index.duplicated().any()

    # axis limits have to be set before we deal with absolute <> data coord transforms
    # xlim can serve to zoom into a plot, or to align axis with other plots going
    # beyond the data range
    # if xlim is not given, we set xlim to the data range
    if xlim:
        granges_df = granges_gr[xlim[0] : xlim[1]].df
    else:
        granges_df = granges_gr.df
        xlim = granges_df["Start"].min(), granges_df["End"].max()
    ax.set_xlim(xlim)

    # we arbitrarliy set y lim to (0, 1) and then convert absolute heights for
    # different plot features into data coordinates within that range
    ax.set_ylim(0, 1)

    # bottom margin so that text does not align with y axis spline
    # exon rects and arrows were slightly cut at the top in plot_gene_model;
    # did not investigate
    # further yet, just also added margin at the top
    ymargin_bottom_and_top_spacer_in = 0.1 / 2.54

    if show_names:

        _, max_text_height_in = _get_max_text_height_width_in_x(
            labels=granges_df["name"],
            fontsize=label_fontsize,
            ax=ax,
            x="inch",
        )
        _, max_text_height_data_coords = _get_max_text_height_width_in_x(
            labels=granges_df["name"],
            fontsize=label_fontsize,
            ax=ax,
            x="data_coordinates",
        )

        size_of_one_track_row_with_spacer_in = (
            max_text_height_in
            + space_between_label_and_patch_in
            + patch_height_in
            + space_between_rows_in
        )

    else:
        size_of_one_track_row_with_spacer_in = patch_height_in + space_between_rows_in

    itvls_with_labels = add_label_positions_to_intervals(
        itvls=granges_df, ax=ax, xlim=xlim, fontsize=label_fontsize
    )

    # maps itvl indices (from granges df) to row indices, starting at 0
    rows_ser = compute_row_placement_for_vertical_interval_dodge(
        intervals=itvls_with_labels,
    )
    n_rows = rows_ser.max() + 1

    # not used here, but kept here because the get_height function is copy-pasted from here atm
    axes_height_in = (
        ymargin_bottom_and_top_spacer_in
        + size_of_one_track_row_with_spacer_in * n_rows
        - space_between_rows_in
        + ymargin_bottom_and_top_spacer_in
    )

    # map itvl_idx -> ylow of rectangle patch
    itvl_ylow = pd.Series(dtype='f4')
    for itvl_idx, row_idx in rows_ser.iteritems():
        itvl_ylow.loc[itvl_idx] = coutils.convert_inch_to_data_coords(
            size=(
                ymargin_bottom_and_top_spacer_in
                + size_of_one_track_row_with_spacer_in * (row_idx + 1)
                - space_between_rows_in
                - patch_height_in
            ),
            ax=ax,
        )[1]
    assert itvl_ylow.lt(1).all()

    patch_height_data_coords = coutils.convert_inch_to_data_coords(
        size=patch_height_in, ax=ax
    )[1]
    label_patch_spacer_height_data_coords = coutils.convert_inch_to_data_coords(
        size=space_between_label_and_patch_in, ax=ax
    )[1]

    # %%
    rectangles = []
    texts = []
    xs = []
    ys = []
    for itvl_idx, row_ser in granges_df.iterrows():  # type: ignore

        if palette:
            curr_color = palette.get(row_ser["name"], color)
        else:
            curr_color = color

        rect = mpatches.Rectangle(
            xy=(row_ser.Start, itvl_ylow.loc[itvl_idx]),
            width=row_ser.End - row_ser.Start,
            height=patch_height_data_coords,
            linewidth=0,
            color=curr_color,
        )
        rectangles.append(rect)

        if show_names:
            # name is not required for all intervals
            if row_ser["name"]:
                # place text in middle of displayed region,
                # not in middle of full interval,
                # parts of which may lie outside of displayed region
                x = min(row_ser["End"], xlim[1]) - (
                    (min(row_ser["End"], xlim[1]) - max(row_ser["Start"], xlim[0])) / 2
                )
                # improve:
                # some names may have less height than others, eg. may be completely lowercase while others are uppercase
                # some names may have 'pg' characters, some may not
                # currently: align everything at baseline
                ax.text(
                    x=x,
                    y=itvl_ylow[itvl_idx]
                    - label_patch_spacer_height_data_coords
                    - max_text_height_data_coords,
                    s=row_ser["name"],
                    ha="center",
                    va="baseline",
                    size=label_fontsize,
                )

    # jj snippets
    ax.add_collection(
        matplotlib.collections.PatchCollection(
            rectangles, match_original=True, zorder=3
        ),
    )
    # %%

    # axis handling
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)

    _format_axis_with_offset_or_order_of_magnitude(ax, offset, order_of_magnitude)

    if title:
        ax.annotate(
            title,
            xy=(1.02, itvl_ylow.max() + patch_height_data_coords),
            xycoords="axes fraction",
            rotation=0,
            ha="left",
            va="top",
            size=title_size,
        )

    # old / scratch
    # =============
    #
    # jj snippets
    # if do_adjust_text:
    #     print('adjusting text')
    #     adjust_text(
    #         texts,
    #         # x=xs,
    #         # y=[1 - ax_fraction_for_rectangles] * len(xs),
    #         # add_objects=None,
    #         ax=ax,
    #         expand_text=(1.2, 1.2),
    #         # expand_points=(1.05, 1.2),
    #         # expand_objects=(1.05, 1.2),
    #         # expand_align=(1.05, 1.2),
    #         autoalign=False,
    #         va="top",
    #         ha="center",
    #         force_text=(0.2, 0.25),
    #         # force_points=(0.2, 0.5),
    #         # force_objects=(0.1, 0.25),
    #         # lim=500,
    #         # precision=0.01,
    #         only_move={"text": "x"},
    #         # text_from_text=True,
    #         # text_from_points=True,
    #         arrowprops=dict(arrowstyle='-', color='black', lw=0.5),
    #     )

def plot_genomic_region_track_get_height(
    granges_gr,
        axes_width,
        xlim,
    show_names=False,
    space_between_label_and_patch_in=0.1 / 2.54,
    space_between_rows_in=0.1 / 2.54,
    patch_height_in=0.4 / 2.54,
    label_fontsize: Optional[float] = None,
        ):

    # height is irrelevant, we just get text height from axes transform which works independent of height
    # width is relevant because we need to vertically dodge intervals with overlapping labels
    fig, ax = plt.subplots(1, 1, figsize=(axes_width, 2), dpi=180)
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)

    # arg handling
    if label_fontsize is None:
        label_fontsize = mpl.rcParams["xtick.labelsize"]
    # granges must have unique index for operations in this function
    assert not granges_gr.df.index.duplicated().any()

    # axis limits have to be set before we deal with absolute <> data coord transforms
    # xlim can serve to zoom into a plot, or to align axis with other plots going
    # beyond the data range
    # if xlim is not given, we set xlim to the data range
    if xlim:
        granges_df = granges_gr[xlim[0] : xlim[1]].df
    else:
        granges_df = granges_gr.df
        xlim = granges_df["Start"].min(), granges_df["End"].max()
    ax.set_xlim(xlim)

    # we arbitrarliy set y lim to (0, 1) and then convert absolute heights for
    # different plot features into data coordinates within that range
    ax.set_ylim(0, 1)

    # bottom margin so that text does not align with y axis spline
    # exon rects and arrows were slightly cut at the top in plot_gene_model;
    # did not investigate
    # further yet, just also added margin at the top
    ymargin_bottom_and_top_spacer_in = 0.1 / 2.54

    if show_names:

        _, max_text_height_in = _get_max_text_height_width_in_x(
            labels=granges_df["name"],
            fontsize=label_fontsize,
            ax=ax,
            x="inch",
        )

        size_of_one_track_row_with_spacer_in = (
            max_text_height_in
            + space_between_label_and_patch_in
            + patch_height_in
            + space_between_rows_in
        )

    else:
        size_of_one_track_row_with_spacer_in = patch_height_in + space_between_rows_in

    itvls_with_labels = add_label_positions_to_intervals(
        itvls=granges_df, ax=ax, xlim=xlim, fontsize=label_fontsize
    )

    # maps itvl indices (from granges df) to row indices, starting at 0
    rows_ser = compute_row_placement_for_vertical_interval_dodge(
        intervals=itvls_with_labels,
    )
    n_rows = rows_ser.max() + 1

    # not used here, but kept here because the get_height function is copy-pasted from here atm
    axes_height_in = (
        ymargin_bottom_and_top_spacer_in
        + size_of_one_track_row_with_spacer_in * n_rows
        - space_between_rows_in
        + ymargin_bottom_and_top_spacer_in
    )

    return axes_height_in


def _get_max_text_height_width_in_x(labels, fontsize, ax, x):
    widths, heights = list(
        zip(
            *[
                coutils.get_text_width_height_in_x(
                    s=s,
                    fontsize=fontsize,
                    ax=ax,
                    x=x,
                )
                for s in labels
            ]
        )
    )
    max_text_width_data_coords = max(widths)
    max_text_height_data_coords = max(heights)
    return max_text_width_data_coords, max_text_height_data_coords


def add_label_positions_to_intervals(
    itvls: pd.DataFrame, ax, xlim, fontsize
) -> pd.DataFrame:
    """

    Parameters
    ----------
    itvls
        Start, End
        name

    Returns
    -------
    Dataframe with label boundaries
        Start, End, center
            interval bounds
        interval_label_start, interval_label_end, interval_label_center
            interval label bounds, maybe inside or outside of interval bounds
        **other cols
    """

    # Setting axis labels MUST be done before determining label sizes in next step (?)
    assert ax.get_xlim() == xlim
    xmin, xmax = xlim

    itvls["label_size_data_coords"] = itvls["name"].apply(
        lambda x: coutils.get_text_width_height_in_x(
            s=x,
            fontsize=fontsize,
            ax=ax,
            x="data_coordinates",
        )[0]
    )
    # interval label will be placed at the center
    # of the interval line initially
    itvls["center"] = itvls["Start"] + (itvls["End"] - itvls["Start"]) / 2

    # Calculate label starts and ends if placed at the center of the interval line
    itvls["interval_label_start"] = itvls["center"] - (  # type: ignore
        itvls["label_size_data_coords"] / 2
    )
    itvls["interval_label_end"] = itvls["center"] + (  # type: ignore
        itvls["label_size_data_coords"] / 2
    )

    # interval labels which start below xmin are shifted such that they start at xmin; they will no longer be centered with respect to the interval line
    itvl_label_below_xmin = itvls["interval_label_start"] < xmin
    # make labels start at xmin
    itvls.loc[itvl_label_below_xmin, "interval_label_start"] = xmin
    # make labels end at xmin + label_size_data_coords
    itvls.loc[itvl_label_below_xmin, "interval_label_end"] = (
        xmin + itvls.loc[itvl_label_below_xmin, "label_size_data_coords"]
    )

    # interval labels which end beyond xmax are shifted left such that they end at xmax
    # not that this could in theory lead to the label start lying below xmin
    itvl_label_above_xmax = itvls["interval_label_end"] > xmax
    itvls.loc[itvl_label_above_xmax, "interval_label_start"] = (
        xmax - itvls.loc[itvl_label_above_xmax, "label_size_data_coords"]
    )
    itvls.loc[itvl_label_above_xmax, "interval_label_end"] = xmax

    # calculate the center of the interval label
    # (which is not necessarily the center of the interval line)
    itvls["interval_label_center"] = (
        itvls["interval_label_start"]
        + (itvls["interval_label_end"] - itvls["interval_label_start"]) / 2
    )

    return itvls


def compute_row_placement_for_vertical_interval_dodge(
    intervals: pd.DataFrame,
) -> pd.Series:
    """Dodge (optionally labeled) intervals only vertically

    Run `add_label_positions_to_intervals` prior to calling this function

    - mode 1: place on new line if labels overlap
    - mode 2 (implement later): place on new line if intervals overlap; dodge overlapping labels

      Longer intervals are placed first, smaller gaps are filled with shorter intervals if the next interval in length sorting order cannot be placed.



      Parameters
      ----------
      intervals
          meaningful index // Start interval_label_start interval_label_end length
          Start,End: interval boundaries
          interval_label_start, interval_label_end
              interval label boundaries, may be within or outside of interval
              this can be computed with `add_label_positions_to_intervals`
          length
              interval length (not considering label bounds)

      Returns
      -------
      dictionary mapping interval_ids to row index starting at 0
    """

    assert not {
        "Start",
        "End",
        "interval_label_start",
        "interval_label_end",
    }.difference(set(intervals.columns))

    if "length" not in intervals:
        intervals["length"] = intervals.eval("End - Start")

    intervals_sorted = intervals.sort_values(["Start", "length"])

    rows_ser = pd.Series(dtype="i4")
    itvls_to_be_placed_in_curr_iter = intervals_sorted.index.tolist()
    interval_to_be_placed_in_next_iter = []
    current_row = 0
    current_x_end = 0
    while itvls_to_be_placed_in_curr_iter:
        for interval_id in itvls_to_be_placed_in_curr_iter:
            interval_ser = intervals_sorted.loc[interval_id]
            if interval_ser[["Start", "interval_label_start"]].min() < current_x_end:  # type: ignore
                # start lies within already covered area of the current row
                interval_to_be_placed_in_next_iter.append(interval_id)
            else:
                rows_ser.loc[interval_id] = current_row
                # update the area covered in the current row
                current_x_end = interval_ser[["End", "interval_label_end"]].max()  # type: ignore
        itvls_to_be_placed_in_curr_iter = interval_to_be_placed_in_next_iter
        interval_to_be_placed_in_next_iter = []
        current_row += 1
        current_x_end = 0

    # flip the order, so that the longest transcript is at the highest row index
    rows_ser = -(rows_ser - rows_ser.max())
    return rows_ser
