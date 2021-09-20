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


_REGION_BOUNDARY_BOX_BASE_PARAMS = {"color": "blue", "alpha": 0.2, "zorder": 0}
_REGION_BOUNDARY_LINE_BASE_PARAMS = dict(color="gray", linewidth=0.5, linestyle="--")


def bar_plot(
    beta_values: pd.DataFrame,
    axes: Union[List, np.ndarray],
    # subject order, colors
    subject_order: Optional[List[str]] = None,
    palette: Union[str, Dict[str, str]] = "Set1",
    # bar plot
    minimum_bar_width_pt: float = 1,
    barplot_lw: float = 0.3,
    merge_overlapping_bars: bool = False,
    # line plot
    show_splines: bool = False,
    # xticks, xlabel
    n_xticklabels=4,
    xlim: Optional[Tuple[float, float]] = None,
    xticks: Optional[Union[List[float], np.ndarray, Tuple[float, ...]]] = None,
    order_of_magnitude: Optional[int] = None,
    offset: Optional[Union[float, bool]] = None,
    xlabel: Optional[str] = "Position (bp)",
    # yticks, ylabel
    ylim: Optional[Tuple[float, float]] = (0, 1),
    yticks_major: Optional[Union[List[float], np.ndarray, Tuple[float, ...]]] = None,
    yticks_minor: Optional[Union[List[float], np.ndarray, Tuple[float, ...]]] = None,
    n_yticklabels=4,
    ylabel: Optional[str] = "Methylation (%)",
    # grid
    grid_lw=1,
    grid_color="black",
    # axes title
    axes_title_position: Literal["right", "top"] = "top",
    axes_title_size=6,
    axes_title_rotation=270,
    # mark ROI
    region_properties: Optional[pd.DataFrame] = None,
    region_boundaries: Optional[str] = None,
    region_boundaries_kws: Optional[Dict] = None,
    #
) -> None:
    """Draw bar plots (one row per subject) on List[Axes], optionally with profile lines

    This function is meant to plot methlevels for individual cytosines or (merged) cytosine-motifs (CG, CHG). In a single call, all cytosine (motifs) are expected to have the same size, ie single cytosines and CG motifs cannot be mixed. This could be changed in the future.

    The width of the bars indicating meth levels for individual cytosines can be increased beyond the motif bp width by setting minimum_bar_width_pt. Note: if minimum_bar_width_pt is used (the default), bars for multiple motifs can overlap. In this case, the region containing the 'overlapping' motif positions is binned in bins with width of minimum_bar_width_pt and the average methylation for each bin is shown if `merge_overlapping_bars = True`.

    - the grid is drawn for both yticks_major and yticks_minor, but yticks_minor has no ticklabels

    Parameters
    ----------
    beta_values
        Chromosome Start End subject beta_value [other_col1, ...].
        subject may be dtype str or categorical. if str, subject_order must be given
    axes: Iterable with one Axes per Subject
    show_splines: in addition to bar plots, draw interpolated and smoothed methylation profile lines
    subject_order: required if beta_values.columns is not categorical, to control subject plot order
    palette: either a palette name known to seaborn.color_palette or a dict mapping subject -> RGB color for all subjects
    order_of_magniture
        if specified, forces scientific notations with this oom. you can only specify oom or offset
    offset
        if specified, forces offest notation. you can only specify oom or offset
    region_properties: Dataframe describing the ROIs, with keys Chromosome Start End [other cols], one one row per ROI. The plot may show CpGs outside of the ROIs, and thus beta_values may contain CpGs outside of the ROIs. region_properties specifies the original ROI boundaries, and is required when region_boundaries are visualized.
    region_boundaries: if not None, mark region boundaries in the plot. Possible values: 'box' -> draw a rectangle patch around the ROI. 'vlines': draw vertical lines
    region_boundaries_kws: passed to the function creating the region boundary visualization, eg patches.Rectangle or Axes.axvline
    xticks: if not None, overrides n_xticklabels
    yticks_major: if not None, overrides n_yticklabels
    yticks_minor: if yticks_major is not None, overrides n_yticklabels
    minimum_bar_width_pt
        minimum bar width for a single motif in pt
    merge_overlapping_bar
        if true, replace clusters of overlapping bars with binned average methylation levels with bin size minimum_bar_width_pt (useful because if minimum_bar_width_pt is used, bars may overlap)
    """

    # assert equal motif size for all motifs
    assert beta_values.eval("End - Start").nunique() == 1

    beta_values = beta_values.copy()

    motif_size = beta_values.iloc[[0]].eval("End - Start").iloc[0]

    beta_values.Chromosome = beta_values.Chromosome.astype(str)

    # prepare/assert args and derived params
    merged_region_boundaries_kws = _process_region_boundary_params(
        region_boundaries, region_boundaries_kws, region_properties
    )

    beta_values = _prepare_beta_values(beta_values, subject_order)

    # we don't use a seaborn function here, so palette preprocessing must be done manually if necessary
    if isinstance(palette, str):
        palette = dict(
            zip(subject_order, sns.color_palette(palette, len(subject_order)))
        )
    else:
        assert isinstance(palette, dict)

    # %%
    # x limits and ticks
    for ax in axes:
        ax.margins(x=minimum_bar_width_pt)
    if xticks is None:
        for ax in axes:
            ax.xaxis.set_major_locator(
                ticker.MaxNLocator(
                    # can get nicer distribution that way and having more ticks which fit is never bad
                    nbins=n_xticklabels,
                    min_n_ticks=n_xticklabels,
                    # steps=np.arange(1, 11),
                    integer=True,
                    prune=None,
                )
            )
    else:
        for ax in axes:
            ax.set_xticks(xticks)
    if yticks_major is None:
        for ax in axes:
            ax.yaxis.set_major_locator(
                ticker.MaxNLocator(
                    # can get nicer distribution that way and having more ticks which fit is never bad
                    nbins=n_yticklabels,
                    min_n_ticks=n_yticklabels,
                    # steps=np.arange(1, 11),
                    integer=True,
                    prune=None,
                )
            )
    else:
        for ax in axes:
            ax.set_yticks(yticks_major)
            ax.set_yticks(yticks_minor, minor=True)
    if ylim is not None:
        for ax in axes:
            ax.set_ylim(ylim)
    if xlim is None:
        xlim = (beta_values["Start"].min(), beta_values["End"].max())
    for ax in axes:
        ax.set_xlim(xlim)

    plt.setp(axes, ylabel=ylabel)

    for ax in axes:
        ax.tick_params(axis="y", which="minor", labelleft=False)

    for ax in axes:
        sns.despine(ax=ax)
        ax.grid(True, which="both", axis="y", lw=grid_lw, c=grid_color, zorder=0)
        ax.set_axisbelow(True)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    # %%

    # Create bar plots (one subject per Axes), with optional spline lines

    # get integer bar width, make it even so that we can extend bar evenly on both sites of CpG dimer
    if minimum_bar_width_pt is not None:
        fig = axes[0].figure
        bar_width_coords = (
            axes[0]
            .transData.inverted()
            .transform(
                fig.dpi_scale_trans.transform(
                    [(0, 0), (minimum_bar_width_pt * 1 / 72, 0)]
                )
            )
        )
        min_bar_width_bp = max(
            bar_width_coords[1, 0] - bar_width_coords[0, 0], motif_size
        )
    else:
        min_bar_width_bp = motif_size

    if min_bar_width_bp and merge_overlapping_bars:

        beta_values_w_bars = beta_values.assign(
            Start_orig=lambda df: df.Start,
            End_orig=lambda df: df.End,
            center=lambda df: df.Start + (motif_size - 1) / 2,
            Start=lambda df: df.Start - (min_bar_width_bp / 2),
            End=lambda df: df.End + (min_bar_width_bp / 2),
        )

        clustered_unique_intervals = (
            pr.PyRanges(
                beta_values_w_bars[
                    ["Chromosome", "Start", "End", "Start_orig", "End_orig", "center"]
                ].drop_duplicates(subset=["Start", "End"])
            )
            .cluster()
            .df
        )

        def bin_motif_group(group_df, min_bar_width_bp):
            if group_df.shape[0] > 1:
                # group_df = clustered_unique_intervals
                pd.testing.assert_frame_equal(group_df.sort_values("Start"), group_df)
                cluster_width_bp = (
                    group_df.iloc[-1]["End_orig"] - group_df.iloc[0]["Start_orig"]
                )
                n_bins = int(np.ceil(cluster_width_bp / min_bar_width_bp))
                total_bins_width = min_bar_width_bp * n_bins
                bins_start = group_df.iloc[0]["Start_orig"] - max(
                    (total_bins_width - cluster_width_bp) / 2, cluster_width_bp * 0.01
                )
                bins_end = group_df.iloc[-1]["End_orig"] + max(
                    (total_bins_width - cluster_width_bp) / 2, cluster_width_bp * 0.01
                )
                bins = np.linspace(bins_start, bins_end, n_bins + 1)
                bin_centers = (bins + min_bar_width_bp / 2)[:-1]
                group_df["center"] = pd.cut(
                    group_df["center"], bins=bins, labels=bin_centers
                )
            return group_df

        clustered_unique_intervals_w_center = (
            clustered_unique_intervals.groupby("Cluster")
            .apply(bin_motif_group, min_bar_width_bp=min_bar_width_bp)
            .assign(Start=lambda df: df["Start_orig"], End=lambda df: df["End_orig"])
            .drop(["Start_orig", "End_orig"], axis=1)
        )

        beta_values_w = beta_values.merge(
            clustered_unique_intervals_w_center,
            on=["Chromosome", "Start", "End"],
            how="left",
        )
        assert beta_values_w["Cluster"].notnull().all()
        # beta_values_w['Chromosome'].isnull().any()

        # Note: multiple motif positions may have the same center; the full info is in clustered_unique_intervals_w_center

        beta_values_w2 = (
            beta_values_w.groupby(
                ["subject", "Chromosome", "Cluster", "center"],
                sort=True,
                observed=True,
            )["beta_value"]
            .mean()
            .reset_index()
            .drop(["Cluster"], axis=1)
        )

    else:
        beta_values_w2 = beta_values
        beta_values_w2["center"] = beta_values_w2.Start + (motif_size - 1) / 2
        clustered_unique_intervals_w_center = beta_values_w2[
            ["Chromosome", "Start", "End", "center"]
        ].drop_duplicates()

    # Must be done AFTER optional beta value aggregation
    if show_splines:

        spline_dfs = []
        for subject, subject_df in beta_values_w2.groupby("subject"):
            spline_dfs.append(
                _smoothed_monotonic_spline2(
                    beta_value_ser=subject_df["beta_value"],
                    pos_arr=subject_df["center"].to_numpy(),
                    subject=subject,
                ).set_index("subject")
            )
        beta_value_lines = pd.concat(spline_dfs)

    for i, (subject, group_df) in enumerate(
        beta_values_w2.groupby("subject", sort=True)
    ):
        axes[i].bar(
            x=group_df["center"],
            height=group_df["beta_value"],
            width=min_bar_width_bp,
            align="center",
            color=palette[subject],
            edgecolor="white",
            linewidth=barplot_lw,
            zorder=1,
        )
        if show_splines:
            view = beta_value_lines.loc[subject]
            axes[i].plot(
                view["Start"], view["beta_value"], color=palette[subject], zorder=2
            )

    # Add subject name as right margin title
    for subject, ax in zip(subject_order, axes):
        if axes_title_position == "top":
            ax.set_title(subject)
        elif axes_title_position == "right":
            ax.annotate(
                subject,
                xy=(1.02, 0.5),
                xycoords="axes fraction",
                rotation=axes_title_rotation,
                ha="left",
                va="center",
                size=axes_title_size,
            )
        else:
            raise NotImplementedError()

    # visualize region boundaries as Rectangle patch or with vlines
    if region_boundaries:

        # motifs may have been moved during binning or jittering of overlapping motif bar clusters
        # adjust region_properties to align with the potential new motif positions
        # beta_values_w2 contains original start and End coords and the (potentially shifted) center, this center is what we want to base our plot boundaries on
        # we use beta_values_w2 because it is always there also without any motif shifting or binning performed


        region_properties = pd.merge(
            region_properties.merge(
                clustered_unique_intervals_w_center[["Chromosome", "Start", "center"]],
                on=["Chromosome", "Start"],
            ).rename(columns={"center": "center_start"}),
            region_properties.merge(
                clustered_unique_intervals_w_center[["Chromosome", "End", "center"]],
                on=["Chromosome", "End"],
            ).rename(columns={"center": "center_end"}),
            on=["Chromosome", "Start", "End"],
        )

        # %%
        # currently, we assume that all motifs in the plot have the same size
        # we plot a bar centered on the motif center with width min_bar_width_bp
        for idx, region_properties_ser in region_properties.iterrows():
            for ax in axes:
                if region_boundaries == "box":

                    # find the motif center of the leftmost motif, then substract half the bar width
                    window_start_x = (
                        region_properties_ser["center_start"] - min_bar_width_bp / 2
                    )

                    # find the motif center of the rightmost motif, then add half the bar width
                    window_end_x = (
                        region_properties_ser["center_end"] + min_bar_width_bp / 2
                    )

                    window_width = window_end_x - window_start_x

                    ax.add_patch(
                        patches.Rectangle(
                            # TODO: improve this hack
                            xy=(window_start_x, ax.get_ylim()[0]),
                            width=window_width,
                            # TODO: improve this hack
                            height=ax.get_ylim()[1] - ax.get_ylim()[0],
                            **merged_region_boundaries_kws,
                        )
                    )
                else:
                    for pos in (
                        region_properties_ser["Start"] + (motif_size - 1) / 2,
                        region_properties_ser["End"]
                        - motif_size
                        - (motif_size - 1) / 2,
                    ):
                        ax.axvline(pos, **merged_region_boundaries_kws)
        # %%

    last_ax = axes[-1]
    # currently bug - does not remove trailing zeros from offset
    # last_ax.xaxis.set_major_formatter(mticker.ScalarFormatterQuickfixed(useOffset=True))
    last_ax.xaxis.set_major_formatter(coutils.ScalarFormatterQuickfixed(useOffset=True))
    if offset and isinstance(offset, bool):
        offset = coutils.find_offset(ax=last_ax)
    if offset:
        last_ax.ticklabel_format(axis="x", useOffset=offset)
    if order_of_magnitude:
        last_ax.ticklabel_format(
            axis="x", scilimits=(order_of_magnitude, order_of_magnitude)
        )

    offset_text_size = mpl.rcParams["xtick.labelsize"] - 1
    axes[-1].xaxis.get_offset_text().set_size(offset_text_size)

    # shift the xlabel position, because the offset label is currently not considered by constrained layout
    # note that mpl.rcParams gives the current rcParams, ie it respects changes made by context managers such as mpl.rc_context
    if xlabel is not None:
        axes[-1].set_xlabel(
            xlabel, labelpad=offset_text_size + mpl.rcParams["axes.labelsize"]
        )


def _process_region_boundary_params(
    region_boundaries, region_boundaries_kws, region_properties
) -> Optional[Dict]:
    """Select default params based on region_boundary type and merge with user-overrides

    Depending on the value of region_boundaries ('box', 'vlines'), different default parameters
    are merged with the user-defined region_boundaries_kws.

    Args:
        region_boundaries: 'box', 'vlines', None
        region_boundaries_kws: passed to function drawing the region boundaries
        region_properties: specify Chromosome, Start, End for the ROI
    """
    if region_boundaries is not None:
        assert region_properties is not None
        if region_boundaries == "box":
            merged_region_boundary_kws = copy(_REGION_BOUNDARY_BOX_BASE_PARAMS)
            if region_boundaries_kws is not None:
                merged_region_boundary_kws.update(region_boundaries_kws)
        elif region_boundaries == "line":
            merged_region_boundary_kws = copy(_REGION_BOUNDARY_LINE_BASE_PARAMS)
            if region_boundaries_kws is not None:
                merged_region_boundary_kws.update(region_boundaries_kws)
        else:
            raise ValueError
        return merged_region_boundary_kws
    else:
        return None


def _smoothed_monotonic_spline2(
    beta_value_ser: pd.Series, pos_arr, subject
) -> pd.DataFrame:
    """Interpolate (pos, beta_value) points with a smoothed cubic monotonic spline

    The monotonic spline creates natural connection lines between the (pos, beta_value)
    points. However, the transitions are not necessarily smooth, which makes visual assessment of
    the plot more difficult. To improve  this, the splines are smoothed with a small gaussian kernel.

    Args:
        beta_value_df: must have columns Start (CpG watson position) and beta_value. Other columsn are ignored.

    Returns:
        pd.DataFrame, with predicted beta values for every bp between the start and end of the region,
        columns: Start, subject, beta_value
    """

    # TODO: improve
    # fill nan positions with linear intrapolation (and extrapolation if necessary)
    # note that extrapolation with scipy.interpolate.interp1d only fills the closest default value
    if beta_value_ser.isna().any():
        print(
            f"WARNING: input data contain NA values for {subject}. Using 1D interpolation prior to spline computation. Especially if NAs are present at the region boundaries, this can distort the profile."
        )
        beta_value_ser = beta_value_ser.interpolate("linear", fill_value="extrapolate")

    # CpG positions are the observed /xin/ positions used to construct the spline
    spline = interpolate.PchipInterpolator(
        pos_arr, beta_value_ser.values, extrapolate=False
    )
    # beta value predictions are computed for every single base between the start and end point (/xout/)
    xout = np.arange(pos_arr[0], pos_arr[-1] + 1)
    beta_pred = spline(xout)

    # the spline-based predictions are smoothed with small gaussian kernel to avoid sharp edges in the profile lines
    gaussian_kernel = scipy.stats.norm.pdf(np.arange(-10, 11), 0, 3)
    # need to normalize to 1
    gaussian_kernel /= np.sum(gaussian_kernel)
    beta_smoothed = convolve(beta_pred, gaussian_kernel, boundary="extend")

    return pd.DataFrame(
        dict(
            Start=xout,
            beta_value=beta_smoothed,
            subject=subject,
        )
    )


def _smoothed_monotonic_spline(beta_value_df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate (pos, beta_value) points with a smoothed cubic monotonic spline

    The monotonic spline creates natural connection lines between the (pos, beta_value)
    points. However, the transitions are not necessarily smooth, which makes visual assessment of
    the plot more difficult. To improve  this, the splines are smoothed with a small gaussian kernel.

    Args:
        beta_value_df: must have columns Start (CpG watson position) and beta_value. Other columsn are ignored.

    Returns:
        pd.DataFrame, with predicted beta values for every bp between the start and end of the region,
        columns: Start, subject, beta_value
    """

    # TODO: improve
    # fill nan positions with linear intrapolation (and extrapolation if necessary)
    # note that extrapolation with scipy.interpolate.interp1d only fills the closest default value
    if beta_value_df["beta_value"].isna().any():
        print(
            f"WARNING: input data contain NA values for {beta_value_df.name}. Using 1D interpolation prior to spline computation. Especially if NAs are present at the region boundaries, this can distort the profile."
        )
        beta_value_df["beta_value"] = beta_value_df["beta_value"].interpolate(
            "linear", fill_value="extrapolate"
        )

    # CpG positions are the observed /xin/ positions used to construct the spline
    xin = beta_value_df["Start"].values
    spline = interpolate.PchipInterpolator(
        xin, beta_value_df["beta_value"].values, extrapolate=False
    )
    # beta value predictions are computed for every single base between the start and end point (/xout/)
    xout = np.arange(xin[0], xin[-1] + 1)
    beta_pred = spline(xout)

    # the spline-based predictions are smoothed with small gaussian kernel to avoid sharp edges in the profile lines
    gaussian_kernel = scipy.stats.norm.pdf(np.arange(-10, 11), 0, 3)
    # need to normalize to 1
    gaussian_kernel /= np.sum(gaussian_kernel)
    beta_smoothed = convolve(beta_pred, gaussian_kernel, boundary="extend")

    return pd.DataFrame(
        dict(
            Start=xout,
            beta_value=beta_smoothed,
            subject=beta_value_df.iloc[0]["subject"],
        )
    )


# gaussian kernel leads to snaking lines
# box kernel leads to sharp edges at minimum of the curve
# use trapezoid kernel instead
# kernel = ac.Trapezoid1DKernel(3, 0.1)
def _lowess_smoother(beta_value_df: pd.DataFrame, smooth=True) -> pd.DataFrame:
    """lowess smoothing for use in groupby call

    missing values will be dropped
    output will be sorted by x-coord

    Args:
        beta_value_df:
    """
    exog = beta_value_df["Start"].values
    arr = lowess(
        endog=beta_value_df["beta_value"].values,
        exog=exog,
        frac=0.1,
        it=20,
        delta=0.01 * (np.max(exog) - np.min(exog)),
        missing="drop",
        return_sorted=True,
    )

    # CpG positions are the observed /xin/ positions used to construct the spline
    xin = arr[:, 0]
    spline = interpolate.PchipInterpolator(xin, arr[:, 1], extrapolate=False)
    # beta value predictions are computed for every single base between the start and end point (/xout/)
    xout = np.arange(xin[0], xin[-1])
    beta_pred = spline(xout)

    if smooth:
        region_size = beta_value_df["Start"].max() - beta_value_df["Start"].min()
        kernel_size = np.round(0.1 * region_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = int(kernel_size)

        # trapezoid kernel
        # half_kernel_size = int((kernel_size-1) / 2)
        # kernel = np.concatenate(
        #         [np.linspace(0, 1, half_kernel_size), [1], np.linspace(0, 1, half_kernel_size)[::-1]]
        # )

        # gaussian_kernel = scipy.stats.norm.pdf(np.arange(-10, 11), 0, 3)
        # need to normalize to 1
        # gaussian_kernel /= np.sum(gaussian_kernel)

        # mean kernel
        kernel = np.ones(kernel_size) / kernel_size
        beta_smoothed = convolve(beta_pred, kernel, boundary="extend")
    else:
        beta_smoothed = beta_pred

    # # the spline-based predictions are smoothed with small gaussian kernel to avoid sharp edges in the profile lines
    # if smooth:
    #     kernel_size = np.round(0.05 * len(beta_value_df))
    #     if kernel_size % 2 == 0:
    #         kernel_size += 1
    #     kernel_size = int(kernel_size)
    #
    #     # # trapezoid kernel
    #     # half_kernel_size = int((kernel_size-1) / 2)
    #     # _lowess_kernel = np.concatenate(
    #     #         [np.linspace(0, 1, half_kernel_size), [1], np.linspace(0, 1, half_kernel_size)[::-1]]
    #     # )
    #
    #     # mean kernel
    #     # _lowess_kernel = np.ones(kernel_size) / kernel_size
    #
    #     # gaussian kernel
    #     # _lowess_gaussian_kernel = scipy.stats.norm.pdf(np.arange(-10, 11), 0, 3)
    #     # # need to normalize to 1
    #     # _lowess_gaussian_kernel /= np.sum(_lowess_gaussian_kernel)
    #
    #     # kernel based convolution
    #     # beta_smoothed = convolve(arr[:, 1], _lowess_kernel, boundary="extend")
    #
    #     # median filter
    #     beta_smoothed = medfilt(arr[:, 1], kernel_size)
    # else:
    #     beta_smoothed = arr[:, 1]

    return pd.DataFrame(
        dict(
            Start=xout,
            beta_value=beta_smoothed,
            subject=beta_value_df.iloc[0]["subject"],
        )
    )


def _prepare_beta_values(
    beta_values: pd.DataFrame, subject_order: List[str]
) -> pd.DataFrame:
    """Convert beta_values['subject'] into Categorical with subject_order if necessary

    Args:
        beta_values: long format, tidy df as used by the Axes-level plotting funcs. NOT the wide format df for the full dataset
        subject_order: list of subject names in desired order

    Returns:
        copy of beta_values, with categorical subject column, sorted by ['subject', 'Start']
    """

    beta_values = beta_values.copy()

    # Assert that this is a tidy, ROI-specific beta value df
    assert beta_values["Chromosome"].nunique() == 1
    assert "subject" in beta_values

    if beta_values["subject"].dtype.name == "category":
        if subject_order is not None and (
            list(beta_values["subject"].cat.categories) != subject_order
        ):
            beta_values["subject"] = pd.Categorical(
                beta_values["subject"], categories=subject_order, ordered=True
            )
    else:
        assert subject_order is not None
        beta_values["subject"] = pd.Categorical(
            beta_values["subject"], categories=subject_order, ordered=True
        )
    # Sorting by Start is sufficient, because we are looking at a set of CpGs on the same Chromosome
    beta_values = beta_values.sort_values(["subject", "Start"])
    return beta_values
