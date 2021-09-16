import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib as mpl
from methlevels.plot_utils import cm
from methlevels.plots2 import plot_gene_model
from methlevels.plots import bar_plot
import codaplot.utils as coutils
import pyranges as pr
from typing import List, Dict, Tuple


def region_plot(
    beta_values_gr: pr.PyRanges,
    gencode_gr: pr.PyRanges,
    chrom: str,
    start: int,
    end: int,
    subject_order: List[str],
    palette: Dict[str, str],
    anno_plots_abs_sizes: Tuple[float, ...],
    figsize: Tuple[float, float],
):
    """Plot methylation in region with annotations

    beta_values_gr
        cpg methylation levels with CpG coordinates
        one column per subject with beta values in dataframe
    gencode_gr
        genomic ranges representation of genomic regions of interes
        must contain at least these GTF format columns (uppercase): Feature
        Features are filtered for: ["transcript", "exon", "UTR"], other features (e.g. gene are ignored)
        consider using `` for filtering GTF gene annos for primary transcripts where possible

    roi_start, roi_end
        beta_value_gr or gencode_gr may contain intervals outside of the region of interest
        specify region to show in plot; this is also useful for 'zooming in' during interactive plotting
    subject_order
        determines subject order in methlevels plot
    palette
        maps subject_name -> subject_color (hex code)
    anno_plots_abs_sizes
        absolute size in inch for each individual annotation axes as Tuple[float, ...], e.g. (cm(5), cm(3))
    figsize
    """

    plot_df = pd.melt(
        beta_values_gr[chrom, start:end].df,  # type: ignore
        id_vars=["Chromosome", "Start", "End"],
        var_name="subject",
        value_name="beta_value",
    )

    gencode_df = gencode_gr["11", start:end].df.loc[  # type: ignore
            lambda df: df["Feature"].isin(["transcript", "exon", "UTR"])
        ]

    n_main_plots = plot_df.subject.nunique()
    fig, axes, big_ax = _setup_region_plot_figure(
        n_main_plots,
        anno_plots_abs_sizes=anno_plots_abs_sizes,
        figsize=figsize,
    )

    bar_plot(
        beta_values=plot_df,
        axes=axes[:n_main_plots],
        subject_order=subject_order,
        region_boundaries=None,
        palette=palette,
        # ylabel='test',
        ylabel=None,
        show_splines=True,
        axes_title_position="right",
        axes_title_size=6,
        axes_title_rotation=0,
        grid_lw=0.5,
        grid_color="lightgray",
        xlabel="Position (bp)",
        offset=True,
        # xlim=(9857000, 9858000),
        # xticks=(9857000, 9857500, 9858000),
        n_xticklabels=5,
        bar_percent=0.01,
        merge_bars=False,
        ylim=(0, 1),
        yticks_major=(0, 1),
        yticks_minor=(0.5,),
        n_yticklabels=3,
        # region_properties: Optional[pd.DataFrame] = None,
        # region_boundaries_kws: Optional[Dict] = None,
    )

    plot_gene_model(
        df=gencode_df,
        offset=True,
        xlabel="Position (bp)",
        ax=axes[-1],
        roi=(start, end),
        rectangle_height=0.4,
        rectangle_height_utrs=0.2,
        perc_of_axis_between_arrows=0.03,
        arrow_length_perc_of_x_axis_size=0.01,
        arrow_height=0.1,
        gene_label_size=6,
    )

    return fig, axes, big_ax


def _setup_region_plot_figure(n_main_plots, anno_plots_abs_sizes, figsize):

    n_rows = n_main_plots + len(anno_plots_abs_sizes)
    fig = plt.figure(constrained_layout=True, dpi=180, figsize=figsize)
    fig.set_constrained_layout_pads(h_pad=0, w_pad=0.02, hspace=0, wspace=0)

    height_ratio_main_axes = (
        (figsize[1] - sum(anno_plots_abs_sizes)) / n_main_plots / figsize[1]
    )
    height_ratios = [height_ratio_main_axes] * n_main_plots + (
        np.array(anno_plots_abs_sizes) / figsize[1]
    ).tolist()
    assert np.isclose(sum(height_ratios), 1)

    gs = gridspec.GridSpec(n_rows, 1, figure=fig, height_ratios=height_ratios)

    axes = np.array([fig.add_subplot(gs[i]) for i in range(n_rows)])

    big_ax = fig.add_subplot(gs[:n_main_plots], frameon=False)
    big_ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
        labelsize=0,
        length=0,
    )

    coutils.add_margin_label_via_encompassing_big_ax(
        fig=fig,
        axes=axes[:n_main_plots],
        big_ax=big_ax,
        axis="y",
        label="Methylation (%)",
    )

    return fig, axes, big_ax


def restrict_to_primary_transcript_if_multiple_transcripts_are_present(gencode_df):
    """

    Parameters
    ----------
    gencode_df
       expects GTF format columns with names: Chromosome Start End Feature gene_id tag
       tag column contains all tags for the feature (csv)
    """

    gene_ids_with_multiple_transcripts = (
        gencode_df.query('Feature == "transcript"')
        .groupby(["gene_id"])
        .size()
        .loc[lambda ser: ser.gt(1)]
        .index
    )

    is_multi_transcript_gene = gencode_df["gene_id"].isin(
        gene_ids_with_multiple_transcripts
    )

    gencode_df = pd.concat(
        [
            gencode_df.loc[~is_multi_transcript_gene],
            gencode_df.loc[is_multi_transcript_gene].loc[
                lambda df: df["tag"].str.contains("appris_principal").fillna(False)
            ],
        ],
        axis=0,
    )
    return gencode_df
