"""Axes-level and runner functions to visualize methylation in genomic regions

Purpose
-------
This module addresses
- visualization of small genomic regions, e.g. DMRs of a few CpGs with some surrounding CpGs
  - here, standard interpolation/smoothing methods are often suboptimal. Here we use a monotonic cubic spline with subsequent gaussian smoothing to get natural methylation profile lines also in very irregular regions with few CpGs
- visualization of many different subjects in the same plot
  - achieved by different means, eg by combining colors and linestyles for the lineplots, and at the same time facetting into different categories of subjects. By using axes level functions, separate legends can be plotted below each facet, which leads to readable legends even for high numbers of subjects. See demo notebook for examples.

Available plots
---------------
- line plot: cpg positions can be shown with point markers or with vertical lines. Uses optimized interpolation (see above) to get natural profiles even in small DMRs
- bar plot: row-facetted by subject, optionally with smoothed lines
- grid plot: grid (subjects, cpgs), each CpG is a circle, the methylation level is color-encoded

Input
-----
All axes level plot functions take tidy beta_values, ie a long-format dataframe with columns Chromosome Start End subject beta_value, which specifies the CpG methylation for all subjects in the region to be plotted.

The runner function `create_region_plots` takes wide-format beta_values (cpgs, samples). This is more efficient for large datasets.

Shared args between all Axes level functions
--------------------------------------------
The axes level functions share a common 'base interface', and have many additional function specific params
beta_values
region_properties
title
region_boundaries

Next steps
----------

- the region boundary in the line plot should go from bottom to top of the axes (currently, there is some padding around the [0,1] ylims, which is not covered by the rectangle patch
- the bar plot should have cpg position markers in case the beta value is 0 (and thus no bar is shown)
- allow highlighting of multiple ROIs in the same plot (currenly only a single ROI is highlighted)
- prior to spline computation, the _smoothed_monotonic_spline function fills NA values with linear interpolation where possible, and extrapolates by using the nearest defined value. Perhaps this should be improved?

"""
print("reloaded plot_methlevels")

from copy import copy
from typing import Optional, List, Union, Dict, Tuple, Literal

import matplotlib.text as mpltext
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.patches as patches
import re
import matplotlib.ticker as ticker

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
import codaplot.utils as coutils


_REGION_BOUNDARY_BOX_BASE_PARAMS = {"color": "blue", "alpha": 0.2}
_REGION_BOUNDARY_LINE_BASE_PARAMS = dict(color="gray", linewidth=0.5, linestyle="--")


def grid_plot(
    beta_values: pd.DataFrame,
    ax: Axes,
    fig: Figure,
    cmap="magma",
    region_properties: Optional[pd.DataFrame] = None,
    region_boundaries: bool = True,
    markersize: int = 50,
    subject_order: Optional[List[str]] = None,
    highlighted_subjects: Optional[List[str]] = None,
    highlight_color="red",
    title: Optional[str] = None,
    ylabel: str = "Subject",
    xlabel: str = "Position [bp]",
    xticks: Optional[List[int]] = None,
    bp_padding=50,
    cbar_args: Optional[Dict] = None,
    cbar_title: str = "% Methylation",
) -> None:
    """Plot a (samples vs CpGs) grid on Axes, with color-coded beta values

    Args:
        beta_values: long format df, columns: Chromosome Start End subject beta_value [other_col1, ...].
        Columns is a scalar Index of subject names. If columns is not categorical, subject_order must be given.
        ax: single Axes
        fig: must be passed so that colorbar can be drawn appropriately
        markersize: size of the points representing CpGs
        xticks: if None, defaults to [Start, Middle, End] of ROI
        region_properties: Series describing the ROI, with keys Chromosome Start End [other cols]. The plot may show flanks surrounding the ROI, and thus beta_values may contain CpGs outside of the ROI. region_properties specifies the original ROI boundaries, and is required when region_boundaries are visualized.
        subject_order: required if beta_values.columns is not categorical, to control subject plot order
        region_boundaries: whether to mark ROI boundaries with vertical lines
        highlighted_subjects: these subject are highlighted with a transparent patch
        highlight_color: color of the subject highlight patch
        bp_padding: padding at the plot ends (purely aesthetic function)
        cbar_args: passed to fig.colorbar, defaults to dict(shrink=0.6, aspect=20)
        cbar_title: passed to fig.colorbar
        cmap: colormap for encoding beta values
    """

    # Implementation notes
    # - region boundaries could also be visualized by rectangle patch. However, this may interfere with the rectangle-patch based highlighting of individual samples.
    # - a better default for xticks may be [Start ROI_start ROI_end End], ie:
    #   xticks = [beta_values_tidy.iloc[0, 1], region_properties['Start'],
    #             region_properties['End'], beta_values_tidy.iloc[-1, 1]

    # prepare/assert args and derived params
    if cbar_args is None:
        cbar_args = dict(shrink=0.6, aspect=20)
    if region_boundaries:
        assert region_properties is not None
    beta_values = _prepare_beta_values(beta_values, subject_order)
    # y coordinate of the different subjects (used for highlighting patches)
    subject_y = pd.Series(dict(zip(subject_order, range(len(subject_order)))))

    # We want horizontal tick lines (to connect cpgs for the same subject), and only
    # one spine at the bottom
    # note: spine trimming seems to be in conflict with constrained layout? Disabled for now.
    ax.grid(True, which="major", axis="y")
    ax.tick_params(axis="y", length=0)
    sns.despine(ax=ax, bottom=False, left=True)

    # Without padding, the boundary cpgs may be plotted only partially
    xlim = (
        beta_values.iloc[0]["Start"] - bp_padding,
        beta_values.iloc[-1]["Start"] + bp_padding,
    )

    # xticks default: three xticks: start, middle, end
    if xticks is None:
        xticks = np.linspace(beta_values.iloc[0, 1], beta_values.iloc[-1, 1], 3)
    ax.set(xticks=xticks, xlabel=xlabel, ylabel=ylabel, xlim=xlim)
    if title is not None:
        ax.set_title(title)

    # Draw the CpG grid
    sc = ax.scatter(
        beta_values["Start"] + 0.5,
        # categorical scatterplot level order: order of appearance in data
        # -> beta values are expected to be already sorted according to subject order
        # this was done in _prepare_beta_values
        beta_values["subject"],
        c=beta_values["beta_value"],
        marker=".",
        edgecolor="black",
        linewidth=0.5,
        s=markersize,
        cmap=cmap,
        vmin=0,
        vmax=1,
        zorder=10,
    )

    # Draw vlines to mark ROI boundaries
    if region_boundaries:
        for idx, region_properties_ser in region_properties.iterrows():
            for pos in region_properties_ser[["Start", "End"]]:
                ax.axvline(pos, color="gray", linewidth=0.5, linestyle="--")

    # Draw colored rectangle patches around subjects of interest
    if highlighted_subjects is not None:
        ax.barh(
            subject_y[highlighted_subjects],
            xlim[1],
            0.8,
            color=highlight_color,
            alpha=0.3,
        )

    # Add colorbar to figure, specify target axes for constrained_layout
    if fig is not None:
        cbar = fig.colorbar(sc, ax=ax, **cbar_args, label=cbar_title)
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(length=1.5, width=0.5, which="both", axis="both")


def line_plot(
    beta_values: pd.DataFrame,
    ax: Axes,
    legend_order: Optional[List[str]] = None,
    palette: Union[str, Dict[str, str]] = "Set1",
    dashes: Optional[Dict] = None,
    seaborn_lineplot_args: Optional[Dict] = None,
    region_properties: Optional[pd.DataFrame] = None,
    region_boundaries: Optional[str] = "box",
    region_boundaries_kws: Optional[Dict] = None,
    bp_padding=20,
    cpg_marks: Optional[str] = "vlines",
    cpg_marks_plot_kws: Optional[Dict] = None,
    xticks: Optional[List[int]] = None,
    yticks: Optional[List[int]] = None,
    title: Optional[str] = None,
    ylabel: str = "Subject",
    xlabel: str = "% Methylation",
    legend_title: str = "",
    smoother="monotonic_spline",
) -> None:
    """Interpolate, smooth and draw methylation profile lines onto an Axes

    Args:
        beta_values: long format df, columns: Chromosome Start End subject beta_value [other_col1, ...].
        Columns is a scalar Index of subject names. If columns is not categorical, subject_order must be given.
        ax: single Axes to plot on
        legend_order: required if beta_values.columns is not categorical, to control the legend entry order
        palette: either a palette name known to seaborn.color_palette or a dict mapping subject -> RGB color for all subjects
        dashes: optional Dict subject -> linestyle. Currently ignored, will be implemented later.
        seaborn_lineplot_args: further args for seaborn.lineplot. Note that seaborn.lineplot() also passes **kwargs to plt.plot
        region_properties: Dataframe describing the ROIs, with keys Chromosome Start End [other cols], one one row per ROI. The plot may show CpGs outside of the ROIs, and thus beta_values may contain CpGs outside of the ROIs. region_properties specifies the original ROI boundaries, and is required when region_boundaries are visualized.
        region_boundaries: if not None, mark region boundaries in the plot. Possible values: 'box' -> draw a rectangle patch around the ROI. 'vlines': draw vertical lines
        region_boundaries_kws: passed to the function creating the region boundary visualization, eg patches.Rectangle or Axes.axvline
        xticks: if None, defaults to [Start, Middle, End] of ROI
        yticks: if None, defaults to
        bp_padding: padding at the plot ends (purely aesthetic function)
        cpg_marks: how to mark CpG positions in the plot. If None, no visualization of CpG positions. 'vlines' -> vertical lines. 'points' -> individual points for all (subject, cpg) data, in color determined by palette
        cpg_marks_plot_kws: passed to the function creating the cpg mark visualization, ie Axes.vline or seaborn.scatterplot
        smoother: 'monotonic_spline', 'lowess'. monotonic spline better for small regions, lowess better for large regions

    Notes:
        - both CpG positions and the ROI boundaries can optionally be marked with vlines. If both informations are visualized, it is better to choose an alternative visualization for one of them. the ROI can also be marked by a rectangle patch and the CpG positions can also be marked by point markers.
    """

    # prepare/assert args and derived params
    merged_region_boundary_kws = _process_region_boundary_params(
        region_boundaries, region_boundaries_kws, region_properties
    )
    beta_values = _prepare_beta_values(beta_values, legend_order)
    if cpg_marks == "vlines":
        cpg_marks_plot_kws_merged = copy(_REGION_BOUNDARY_LINE_BASE_PARAMS)
    if cpg_marks == "points":
        cpg_marks_plot_kws_merged = dict(s=30, marker=".")
    if cpg_marks_plot_kws is not None and cpg_marks in ["vlines", "points"]:
        cpg_marks_plot_kws_merged.update(cpg_marks_plot_kws)
    if dashes is None:
        dashes = False
    if seaborn_lineplot_args is None:
        seaborn_lineplot_args = {}

    # x limits and ticks
    # Without padding, the boundary cpgs may be plotted only partially
    xlim = (
        beta_values.iloc[0]["Start"] - bp_padding,
        beta_values.iloc[-1]["Start"] + bp_padding,
    )
    # default: three xticks: start, middle, end
    # alternative would be eg first cpg, first roi cpg, last roi cpg, end
    # xticks = [beta_values_tidy.iloc[0, 1], region_properties['Start'],
    #           region_properties['End'], beta_values_tidy.iloc[-1, 1]]
    if xticks is None:
        xticks = np.linspace(beta_values.iloc[0, 1], beta_values.iloc[-1, 1], 3)
    if yticks is None:
        yticks = np.linspace(0, 1, 5)
    ax.set(xticks=xticks, xlabel=xlabel, ylabel=ylabel, xlim=xlim, yticks=yticks)
    if title is not None:
        ax.set_title(title)

    sns.despine(ax=ax)

    if smoother == "monotonic_spline":
        smoothing_func = _smoothed_monotonic_spline
    elif smoother == "lowess":
        smoothing_func = _lowess_smoother
    else:
        raise ValueError("Unknown smoother")
    # connect (subject, pos, beta_value) data with smoothed splines
    print("using smoothing")
    beta_value_lines = beta_values.groupby("subject", group_keys=False).apply(
        smoothing_func,
    )

    # Draw lines with hue and linestyle levels
    sns.lineplot(
        x="Start",
        y="beta_value",
        hue="subject",
        # style="subject",
        # style_order=legend_order,
        # hue_order=legend_order,
        palette=palette,
        # dashes=dashes,
        ax=ax,
        data=beta_value_lines,
        **seaborn_lineplot_args,
    )

    # Make legend created by seaborn nicer
    # - seaborn adds a dummy legend entry at the top to get a title which is aligned with the labels
    # - this makes it difficult to optionally show or remove the legend title
    # - why is this done in seaborn? because of a fontsize issue with the legend title? do i need to continue using this workaround myself?
    # -> for now, remove the dummy label by exchanging the entire legend
    (dummy, *handles), (dummy, *labels) = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), title=legend_title
    )

    # Mark CpG positions with vlines or with markers in the line plot
    if cpg_marks == "vlines":
        first_subject = legend_order[0]
        for pos in beta_values.query("subject == @first_subject")["Start"]:
            ax.axvline(pos, **cpg_marks_plot_kws_merged)
    elif cpg_marks == "points":
        sns.scatterplot(
            x="Start",
            y="beta_value",
            hue="subject",
            palette=palette,
            ax=ax,
            data=beta_values,
            legend=None,
            **cpg_marks_plot_kws_merged,
        )

    # Visualize region boundaries with rectangle patch or with vlines
    if region_boundaries == "box":
        for idx, region_properties_ser in region_properties.iterrows():
            ax.add_patch(
                patches.Rectangle(
                    (region_properties_ser["Start"], 0),
                    (
                        region_properties_ser["End"]
                        - 2
                        - region_properties_ser["Start"]
                        + 1
                    ),
                    1,
                    **merged_region_boundary_kws,
                )
            )
    elif region_boundaries == "line":
        for idx, region_properties_ser in region_properties.iterrows():
            for pos in region_properties_ser[["Start", "End"]]:
                ax.axvline(pos, **merged_region_boundary_kws)


def create_region_plots(
    beta_values: pd.DataFrame,
    cpg_gr: pr.PyRanges,
    regions: pd.DataFrame,
    output_dir,
    plot_types: List[str],
    title_prefix: Optional[str] = None,
    title_col: Optional[str] = None,
    filetypes: Optional[List[str]] = None,
    slack_abs: int = 0,
    subject_order: Optional[List[str]] = None,
    legend_order: Optional[List[str]] = None,
    show_title=True,
    grid_kwargs: Optional[Dict] = None,
    line_kwargs: Optional[Dict] = None,
    bar_kwargs: Optional[Dict] = None,
    bar_plot_facet_height_in: float = 2 / 2.54,
    plot_width_in: float = 8 / 2.54,
    line_plot_height_in: float = 8 / 2.54,
    grid_plot_line_height_in: Optional[float] = None,
    dpi: int = 180,
) -> None:
    """Runner function which creates region plots of different kinds for multiple ROIs

    Args:
        beta_values: (cpgs, samples). Sorted by Grange columns, CpG rows must be unique (ie no duplicates allowed). Columns: scalar index of subject names. if not categorical, subject_order must be given.
        cpg_gr: CpG index aligned with the rows in beta_values, with running element_id metadata column
        regions: Dataframe detailing ROIs to be plotted, columns Chromosome Start End [Strand, {title_col}, ...]. Optionally with column rois, to describe one or more subregions within the full interval which build the set of rois (otherwise the entire region is assumed to be one monolithic ROI). the 'rois' column contains Tuple[Tuple[int, int]], detailing start and end for each region
        output_dir: plots will be put here, under {title}.{suffix} suffices are defined by filetypes, the title depends on title_col arg
        filetypes: list of filetypes, defaults to ['png', 'pdf']
        title_col: name of column in regions df, if given, files are saved as {title_col[region_idx]}_{Chr:Start-End}, otherwise, only the coordinates are used
        slack_abs: enlarge ROI to both sides by slack_abs bp
        subject_order: must be given if the subject index is not categorical and if bar or grid plot is used
        legend_order: must be given if the subject index is not categorical and if line plots are used
        plot_types: one or more from ['grid', 'line', 'bar']
        show_title: whether to pass the title to the plotting functions
        grid_kwargs: passed to ml.grid_plot
        line_kwargs: passed to ml.line_plot
        bar_kwargs: passed to ml.bar_plot
        plot_width_in: used for all plot types. The height is controlled by plot type specific parameters
        bar_plot_facet_height_in: height for a single bar plot facet, used to determine bar_plot figure size. Space for other plot element (ticklabels etc.) is automatically added.
        line_plot_height_in: height for line plot figure
        grid_plot_line_height_in: height of a single grid_plot line, used to compute full figure height. Space for ticks, ticklabels etc. is automatically added. if None, the y-axis label height is used as line height
    """

    # Implementation notes:
    # - the flank size could also be given as percentage of the ROI size
    # - flank size could be different upstream/downstream
    # - the flank size could also be specified per ROI in the dmr df (potentially with different upstream
    # and downstream values)

    # check and process arguments
    if filetypes is None:
        filetypes = ["png", "pdf"]
    if title_col:
        assert title_col in regions
    assert "element_idx" in cpg_gr.df
    if beta_values.columns.nlevels > 1:
        raise NotImplementedError("Currently only implemented for subject-level data")
    if beta_values.columns.get_level_values(0).dtype.name != "category":
        assert subject_order is not None
        # we will pass subject order to the plot functions, which will re-sort the plot data internally
        # so no need to take action here as long as we have the subject_order as arg
    if line_kwargs is None:
        line_kwargs = {}
    if grid_kwargs is None:
        grid_kwargs = {}
    if bar_kwargs is None:
        bar_kwargs = {}

    # Create plots defined by /plot_type/ for each ROI in /regions/
    for idx, region_ser in regions.iterrows():
        print(region_ser)

        # Retrieve the index df [Chromosome Start End element_idx] for all CpGs in the current ROI
        # The /element_idx/ gives the integer index of the corresponding CpGs in the /beta_values/ df
        idx_df = cpg_gr[
            region_ser["Chromosome"],
            (region_ser["Start"] - slack_abs) : (region_ser["End"] + slack_abs),
        ].df.reset_index(drop=True)

        # Retrieve CpG methylation data and bring into tidy format
        # DF [///Chromosome Start End subject beta_value]
        plot_beta_values = (
            beta_values.iloc[idx_df["element_idx"].values]
            .reset_index(drop=True)
            .stack(0, dropna=False)
            .reset_index(-1)
            .set_axis(["subject", "beta_value"], axis=1, inplace=False)
            .join(idx_df.iloc[:, 0:3])
            .reset_index(drop=True)
            .loc[:, ["Chromosome", "Start", "End", "subject", "beta_value"]]
        )

        # Determine the ROI title, which is used in the filename, and optionally as plot title
        # The ROI title specifies the genomic interval, and optionally the value in /title_col/ (from regions DF)

        curr_title = title_prefix + "_" if title_prefix else ""
        curr_title += region_ser[title_col] + " | " if title_col else ""
        curr_title += (
            f"{region_ser['Chromosome']}_{region_ser['Start']}-{region_ser['End']}"
        )

        # region_properties are the full region, unless 'rois' column in present
        if "rois" in region_ser:
            region_properties = pd.DataFrame(
                region_ser["rois"], columns=["Start", "End"]
            ).assign(Chromosome=region_ser["Chromosome"])[
                ["Chromosome", "Start", "End"]
            ]
        else:
            # turn region_ser into dataframe with columns Chromosome, Start, End, ...
            region_properties = region_ser.to_frame().T

        # Now create all plots specified in /plot_types/

        # Args shared by all axes-level plot functions
        common_plot_args = dict(
            beta_values=plot_beta_values,
            region_properties=region_properties,
            title=curr_title if show_title else None,
        )

        if "grid" in plot_types:

            # Figure size estimation
            # TODO: improve figure size estimation
            estimated_colorbar_size = 1.5 / 2.54
            text_height = mpl.rcParams["xtick.labelsize"] * 1 / 72
            if grid_plot_line_height_in is None:
                grid_plot_line_height_in = text_height

            # Create plot
            fig, ax = plt.subplots(
                1,
                1,
                constrained_layout=True,
                figsize=(
                    plot_width_in + estimated_colorbar_size,
                    grid_plot_line_height_in * len(subject_order) + 2 / 2.54,
                ),
                dpi=dpi,
            )
            grid_plot(
                **common_plot_args,
                ax=ax,
                fig=fig,
                subject_order=subject_order,
                **grid_kwargs,
            )

            # Save
            for suffix in filetypes:
                fp = (
                    output_dir + f"/{curr_title.replace(' | ', '_')}_kind-grid.{suffix}"
                )
                print("Saving", fp)
                fig.savefig(fp)

        if "line" in plot_types:

            # Figure size estimation
            # TODO: this should be computed correctly
            estimated_legend_size = 2.5 / 2.54

            # Create plot
            fig, ax = plt.subplots(
                1,
                1,
                constrained_layout=True,
                figsize=(plot_width_in + estimated_legend_size, line_plot_height_in),
                dpi=dpi,
            )
            line_plot(
                **common_plot_args, ax=ax, **line_kwargs, legend_order=legend_order
            )

            # Save
            for suffix in filetypes:
                fp = (
                    output_dir + f"/{curr_title.replace(' | ', '_')}_kind-line.{suffix}"
                )
                print("Saving", fp)
                fig.savefig(fp)

        if "bar" in plot_types:

            # Create plot
            fig, axes = plt.subplots(
                len(subject_order),
                1,
                constrained_layout=True,
                figsize=(plot_width_in, bar_plot_facet_height_in * len(subject_order)),
                dpi=dpi,
            )
            bar_plot(
                **common_plot_args, axes=axes, **bar_kwargs, subject_order=subject_order
            )

            # Save
            for suffix in filetypes:
                fp = output_dir + f"/{curr_title.replace(' | ', '_')}_kind-bar.{suffix}"
                print("Saving", fp)
                fig.savefig(fp)

    # Avoid display of figures in notebooks
    plt.close("all")
