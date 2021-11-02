import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib as mpl
from methlevels.plot_utils import cm
from methlevels.plot_genomic import plot_gene_model, plot_genomic_region_track
from methlevels.plot_methlevels import bar_plot, _get_bar_width_bp
import codaplot.utils as coutils
import pyranges as pr
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

print("reloaded plot_region")


def region_plot(
    beta_values_gr: pr.PyRanges,
    chrom: str,
    start: int,
    end: int,
    subject_order: List[str],
    title: Optional[str] = None,
    bar_plot_kwargs: Optional[Dict[str, Any]] = None,
    gene_axes_size: float = 1 / 2.54,
    anno_axes_size: float = 0.5 / 2.54,
    drop_empty_axes: bool = False,
    fig_width: float = 7 / 2.54,
    fig_height: Optional[float] = None,
    methlevel_axes_size: Optional[float] = None,
    h_pad=0,
    anno_axes_padding: float = 0.02,
    gene_anno_gr: Optional[pr.PyRanges] = None,
    genomic_regions: Optional[Dict[str, pr.PyRanges]] = None,
    genomic_region_is_cyt_bound: Optional[Dict[str, bool]] = None,
    custom_axes_funcs: Optional[Dict[str, Tuple[Callable, Dict[str, Any]]]] = None,
    custom_axes_sizes: Tuple[float, ...] = tuple(),
    gene_model_kwargs: Optional[Dict[str, Any]] = None,
    plot_genomic_region_track_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    offset: Union[float, bool] = True,
    debug=False,
):
    """Plot methylation in region with annotations

    Parameters
    ----------
    chrom, start, end
        start and end provide the xlim
        beta_values_gr, gene_anno_gr and genomic_regions (all individual pyranges) are automatically restricted to chrom, start and end; for performance reason, still consider doing this upfront if you have large pyranges
    beta_values_gr
        cpg methylation levels with CpG coordinates
        one column per subject with beta values in dataframe
        ie Chromosome Start End subject1 subject2 ...
    gencode_gr
        genomic ranges representation of genomic regions of interes
        must contain at least these GTF format columns (uppercase): Feature
        Features are filtered for: ["transcript", "exon", "UTR"], other features (e.g. gene are ignored)
        consider using `` for filtering GTF gene annos for primary transcripts where possible
    bar_plot_kwargs
        passed to bar_plot
        do not use these kwargs, they are managed by this function: ['beta_values', 'axes', 'subject_order', 'offset']

    roi_start, roi_end
        beta_value_gr or gencode_gr may contain intervals outside of the region of interest
        specify region to show in plot; this is also useful for 'zooming in' during interactive plotting
    subject_order
        determines subject order in methlevels plot
    color
        default color for methlevel plots
    h_pad
        passed to constrained_layout. 0 will often be best, argument may be removed in the future
    anno_axes_padding
        padding between anno_axes and between anno_axes and surrounding axes
    palette
        maps subject_name -> subject_color (hex code)
        supersedes color
    gene_axes_size
        size in inch for the gene anno axes (if present)
    anno_axes_size
        size in inch for each individual annotation axes (if present)
    drop_empty_axes
        if True, do not create (empty) axes for empty pyranges
    fig_width
        figure width in inch
    fig_height
        figure height in inch, automatically determine methlevel_axes_size; do not pass both fig_height and methlevel_axes_size
    methlevel_axes_size
        size in inch for each individual methlevels axes; do not pass both fig_height and methlevel_axes_size
    offset
        if specified, forces offest notation.
        either specify True for automatic offset detection or a float
    genomic_regions
        dict track_name -> PyRanges of non-overlapping intervals
    plot_genomic_region_track_args
        dict track_name -> kwargs for plot_genomic_region_tracks
        optional, can also define for some tracks and not for others
    debug
        if True, plot x axis for all plots to make sure they align
    """

    if not bar_plot_kwargs:
        bar_plot_kwargs = {}
    mangaged_bar_plot_args = ["beta_values", "axes", "subject_order", "offset"]
    assert not any([x in bar_plot_kwargs.keys() for x in mangaged_bar_plot_args])

    if not gene_model_kwargs:
        gene_model_kwargs = {}

    if genomic_region_is_cyt_bound is None:
        genomic_region_is_cyt_bound = {}

    pd.testing.assert_frame_equal(
        beta_values_gr[chrom].df.sort_values(["Chromosome", "Start", "End"]),  # type: ignore
        beta_values_gr[chrom].df,  # type: ignore
    )

    plot_df = pd.melt(
        beta_values_gr[chrom, start:end].df,  # type: ignore
        id_vars=["Chromosome", "Start", "End"],
        var_name="subject",
        value_name="beta_value",
    )

    if not plot_genomic_region_track_kwargs:
        plot_genomic_region_track_kwargs = {}

    n_main_plots = plot_df.subject.nunique()
    n_annos = len(genomic_regions) if genomic_regions is not None else 0

    # %%
    fig, axes_d = _setup_region_plot_figure(
        n_main_plots,
        gene_axes_size=gene_axes_size if gene_anno_gr else 0,
        anno_axes_size=anno_axes_size,
        methlevel_axes_size=methlevel_axes_size,
        anno_axes_padding=anno_axes_padding,
        custom_axes_sizes=custom_axes_sizes,
        n_annos=n_annos,
        h_pad=h_pad,
        fig_width=fig_width,
        fig_height=fig_height,
    )

    # assert equal motif size for all motifs
    # add xmargin (in bp) to xlim, xlim will be passed to all plotting functions

    # assert that only one motif size is present
    assert plot_df.eval("End - Start").nunique() == 1
    motif_size = plot_df.iloc[[0]].eval("End - Start").iloc[0]  # type: ignore

    ax = axes_d["methlevels"][0]
    ax.set_xlim(start, end)
    assert "minimum_bar_width_pt" in bar_plot_kwargs
    # add spline line width because axis spline will cover bars if its wide even if there is a margin to show the bar fully in principle
    min_bar_width_bp = _get_bar_width_bp(
        ax=ax,
        minimum_bar_width_pt=bar_plot_kwargs["minimum_bar_width_pt"]
        + mpl.rcParams["axes.linewidth"],
        motif_size=motif_size,
    )
    start = start - min_bar_width_bp
    end = end + min_bar_width_bp

    # Chromosome Start(orig) End(orig) center(bar) Start_bar End_bar ...
    unique_coords = bar_plot(
        beta_values=plot_df,
        axes=axes_d["methlevels"],
        xlim=(start, end),
        xmargin=0,
        subject_order=subject_order,
        offset=offset,
        **bar_plot_kwargs,
    )

    # # Remove x axis labels if there is any annotation plot below (which will provide the coordinates, unless we are debugging this plot
    # if (genomic_regions or gene_anno_gr or custom_axes_funcs) and not debug:
    #     axes_d["methlevels"][-1].tick_params(
    #         axis="x",
    #         which="both",
    #         bottom=False,
    #         labelbottom=False,
    #         labelsize=0,
    #         length=0,
    #     )
    #     axes_d["methlevels"][-1].set_xlabel(None)
    # %%

    if genomic_regions:

        genomic_regions_subsetted = {
            name: gr[chrom, start:end] for name, gr in genomic_regions.items()
        }

        if bar_plot_kwargs.get("merge_overlapping_bars") == "dodge":

            genomic_regions_subsetted2 = {}

            for name, gr in genomic_regions_subsetted.items():
                if genomic_region_is_cyt_bound.get(name):
                    genomic_regions_subsetted2[name] = pr.PyRanges(
                        gr.df.merge(
                            unique_coords[["Chromosome", "Start", "Start_bar"]],
                            on=["Chromosome", "Start"],
                            how="left",
                        )
                        .merge(
                            unique_coords[["Chromosome", "End", "End_bar"]],
                            on=["Chromosome", "End"],
                            how="left",
                        )
                        # we cannot find shifted interval borders if they lie outside of the plotted area; in these cases, just use the original interval boundary
                        .assign(
                            Start_bar=lambda df: df["Start_bar"].mask(
                                df["Start"].lt(start), df["Start"]
                            ),
                            End_bar=lambda df: df["End_bar"].mask(
                                df["End"].gt(end), df["End"]
                            ),
                        )
                        .drop(["Start", "End"], axis=1)
                        .rename(columns={"Start_bar": "Start", "End_bar": "End"})
                    )
                else:
                    genomic_regions_subsetted2[name] = gr

            genomic_regions_subsetted = genomic_regions_subsetted2

        for i, (ax, (track_name, gr)) in enumerate(
            zip(axes_d["annos"], genomic_regions_subsetted.items())
        ):
            if gr.empty:
                print(f"WARNING {track_name} has no intervals in the ROI")
                continue
            if debug:
                no_coords = False
            elif (i == len(genomic_regions) - 1) and (not gene_anno_gr):
                no_coords = False
            else:
                no_coords = True
            plot_genomic_region_track(
                gr,
                ax,
                offset=offset,
                xlim=(start, end),
                ax_abs_height=anno_axes_size,
                label_fontsize=6,
                no_coords=no_coords,
                **plot_genomic_region_track_kwargs.get(track_name, {}),
            )

    if gene_anno_gr:
        gencode_df = gene_anno_gr[chrom, start:end].df.loc[  # type: ignore
            lambda df: df["Feature"].isin(["transcript", "exon", "UTR"])
        ]
        plot_gene_model(
            df=gencode_df,
            offset=offset,
            xlim=(start, end),
            xlabel="Position (bp)",
            ax=axes_d["gene_anno"],
            **gene_model_kwargs,
        )
    # %%

    if custom_axes_funcs:
        for ax, (track_name, (func, kwargs)) in zip(
            axes_d["custom"], custom_axes_funcs.items()
        ):
            ax.set_xlim(start, end)
            func(
                ax=ax,
                context_infos=dict(
                    start=start,
                    end=end,
                    title=title,
                    unique_coords=unique_coords,
                    bar_width=min_bar_width_bp,
                    barplot_lw=bar_plot_kwargs.get("barplot_lw", 0),
                ),
                **kwargs,
            )

    # Remove x axis labels if there is any annotation plot below (which will provide the coordinates, unless we are debugging this plot
    for ax in axes_d["all"][:-1]:
        ax.tick_params(
            axis="x",
            which="both",
            bottom=False,
            labelbottom=False,
            labelsize=0,
            length=0,
        )
        ax.set_xlabel(None)

    if title:
        axes_d["methlevels"][0].set_title(title)

    return fig, axes_d


# %%
def _setup_region_plot_figure(
    n_main_plots,
    gene_axes_size,
    anno_axes_size,
    anno_axes_padding,
    custom_axes_sizes,
    n_annos,
    h_pad,
    fig_width,
    methlevel_axes_size,
    fig_height,
):
    """
    Parameters
    ----------
    n_main_plots
        number of subplots in methlevels plot, one per subject
    gene_axes_size
        size in inch of gene model plot, 0 if not plotted
    anno_axes_size
        size (inch) of each anno subplot, None if not plotted
    methlevel_axes_size, fig_height
        one must be None
    """

    if bool(methlevel_axes_size) + bool(fig_height) != 1:
        raise ValueError()

    n_custom = len(custom_axes_sizes)
    n_anno_paddings = 1 + n_custom + n_annos
    n_rows = (
        n_main_plots
        + 1  # anno_padding
        + n_custom
        + n_custom  # anno_padding
        + n_annos
        + n_annos  # anno_padding
        + bool(gene_axes_size)
    )

    if fig_height is None:
        fig_height = (
            methlevel_axes_size * n_main_plots
            + sum(custom_axes_sizes)
            + n_annos * anno_axes_size
            + gene_axes_size
            + n_anno_paddings * anno_axes_padding
        )
        print(
            'figure height is computed as "methlevel_axes_size * n_main_plots + gene_axes_size + n_annos * (anno_axes_size + anno_axes_padding) - anno_axes_padding", giving',
            fig_height,
            "inch or",
            fig_height * 2.54,
            "cm",
        )
    figsize = fig_width, fig_height

    fig = plt.figure(constrained_layout=True, dpi=180, figsize=figsize)
    fig.set_constrained_layout_pads(h_pad=h_pad, w_pad=0, hspace=0, wspace=0)

    height_ratio_main_axes = (
        (
            figsize[1]
            - sum(custom_axes_sizes)
            - n_annos * anno_axes_size
            - gene_axes_size
            - n_anno_paddings * anno_axes_padding
        )
        / n_main_plots
        / figsize[1]
    )
    anno_axes_ratio = anno_axes_size / figsize[1]
    custom_axes_ratios = (np.array(custom_axes_sizes) / figsize[1]).tolist()
    anno_axes_padding_ratio = anno_axes_padding / figsize[1]
    height_ratios = (
        [height_ratio_main_axes] * n_main_plots
        + [anno_axes_padding_ratio]
        + list(
            itertools.chain.from_iterable(
                zip(custom_axes_ratios, [anno_axes_padding_ratio] * n_custom)
            )
        )
        + [anno_axes_ratio, anno_axes_padding_ratio] * n_annos
        + [gene_axes_size / figsize[1]] * bool(gene_axes_size)
    )
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

    methlevel_axes = axes[:n_main_plots]
    custom_axes = (
        axes[(n_main_plots + 1) : (n_main_plots + n_custom * 2) : 2]
        if n_custom
        else None
    )
    anno_axes = (
        axes[
            (n_main_plots + 1 + n_custom * 2) : (
                n_main_plots + n_custom * 2 + n_annos * 2
            ) : 2
        ]
        if n_annos
        else None
    )
    gene_anno_axes = axes[-1] if gene_axes_size else None
    all_plot_axes_d = {
        "methlevels": methlevel_axes,
        "custom": custom_axes,
        "annos": anno_axes,
        "gene_anno": gene_anno_axes,
        "all": axes,
        "methlevel_ylabel": big_ax,
    }
    all_plot_axes_l = np.concatenate(
        (
            methlevel_axes,
            custom_axes if custom_axes is not None else [],
            anno_axes if anno_axes is not None else [],
            [gene_anno_axes] if gene_anno_axes is not None else [],
        )
    )
    if n_annos:
        # spacer_axes = axes[n_main_plots : (n_main_plots + n_annos * 2 + 1) : 2]
        spacer_axes = [ax for ax in axes if ax not in all_plot_axes_l]
        for ax in spacer_axes:
            coutils.strip_all_axis(ax)
    else:
        spacer_axes = None

    all_axes_d = all_plot_axes_d.copy()
    all_axes_d["spacer"] = spacer_axes
    return fig, all_axes_d


# %%


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
