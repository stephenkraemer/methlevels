from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib as mpl


import pyranges as pr

import pandas as pd
import numpy as np
import pytest

import methlevels as ml
from methlevels.utils import (
    read_csv_with_padding,
)
from methlevels.utils import NamedColumnsSlice as ncls
from methlevels.plots2 import plot_gene_model, get_text_width_data_coordinates
from methlevels.plots import bar_plot

import mouse_hema_meth.styling as mhstyle

mpl.rcParams.update(mhstyle.paper_context)
import mouse_hema_meth.utils as ut
import mouse_hema_meth.paths as mhpaths
import mouse_hema_meth.methylome.alignments_mcalls.get_meth_stats_for_granges_lib as get_meth_stats_for_granges_lib
import mouse_hema_meth.methylome.clustering.characterize_clustering_2_paths as characterize_clustering_2_paths
import mouse_hema_meth.shared_vars as mhvars
import mouse_hema_meth.styling as mhstyle
import matplotlib.gridspec as gridspec
import codaplot.utils as coutils

cm = ut.cm


def test_region_plot2():

    # dms_all = get_meth_stats_for_granges_lib.dmr_meth_stats_all_pops_no_qc_poplevel
    gtf_fp = (
        "/omics/odcf/analysis/OE0219_projects/mouse_hematopoiesis/databases/gene_annotations"
        "/gencode.vM19.annotation.no-prefix.gtf"
    )

    gencode_gr = pr.read_gtf(gtf_fp, duplicate_attr=True)

    roi_chrom = "11"
    roi_start = 19_018_985 - 1000
    roi_end = 19_018_985 + 5000
    # roi_start = 18_879_817
    # roi_end = 19_018_985

    if recompute:

        "tabix /omics/odcf/analysis/OE0219_projects/mouse_hematopoiesis/results/wgbs/results_per_pid/v1_bistro-0.2.0_odcf-alignment/hsc_1/meth/meth_calls/mcalls_hsc_1_CG_chrom-merged_strands-merged.bed.gz 11:18879817-19018985 | wc -l"

        # ===

        from mouse_hema_meth.methylome.alignments_mcalls.meth_calling_paths import (
            ds1_metadata_table_tsv,
        )

        ds1_metadata_table = pd.read_csv(ds1_metadata_table_tsv, sep="\t", header=0)

        hierarchy_bed_calls_ds1 = ml.BedCalls(
            metadata_table=ds1_metadata_table,
            tmpdir=mhpaths.project_temp_dir,
            pop_order=mhvars.ds1.all_pops,
            beta_value_col=6,
            n_meth_col=7,
            n_total_col=8,
        )

        """
                intervals_df: must start with Chromosome, Start, End, region_id.
                    Must be sorted by grange cols. region_id most be monotonic increasing.
                    Chromosome column should be categorical of strings. If it is only str dtype,
                    it will be (internally) converted to categorical, using the given order of chromosomes.
                    The original dataframe remains unchanged.
                    Optionally, annotation columns may be added to the intervals df.
                    They will be added to the results (available through anno df of
                    the returned MethStats object).
                    The interval df may contain overlapping intervals, duplicate intervals and emtpy intervals (containing no CpGs).
                    If the interval df contains duplicate or overlapping genomic intervals (shared CpGs), it must contain
                    one or more additional index variables, so that the complete index
                    if fully unique. These additional index variables must be distinguished
                    from annotation variables by passing them via /additional_index_cols/
                elements: whether to save the result to the elements slots of the resulting MethStats object.
                    This will almost always be wanted - probably remove this option later.
        """

        intervals = pd.DataFrame(
            {"Chromosome": ["11"], "Start": [roi_start], "End": [roi_end], "region_id": [0]}
        )
        n_cores = 12

        get_ms_results_dir = mhpaths.project_temp_dir + "/mlr-test"
        Path(get_ms_results_dir).mkdir(exist_ok=True)
        import methlevels.recipes

        methstats_paths_d = methlevels.recipes.compute_meth_stats(
            bed_calls=hierarchy_bed_calls_ds1,
            intervals=intervals,
            result_dir=get_ms_results_dir,
            filter_args=None,
            root_subject="hsc",
            result_name_prefix="arbitrary",
            result_name_suffix="",
            cores=n_cores,
        )

        # Load pop level methstats
        methstats_pops: ml.MethStats = ut.from_pickle(
            methstats_paths_d["per-subject"]["meth_stats_obj"]
        )
        # assert that no qc filtering was done accidentally
        assert methstats_pops.counts.shape[0] == intervals.shape[0]

        dms_plot = get_meth_stats_for_granges_lib.DmrMethStatsDfs(
            mhpaths.project_temp_dir + "/asfasdfasdf"
        )

        print("save pop level meth stats dfs")
        get_meth_stats_for_granges_lib.methstats_to_dfs(
            methstats=methstats_pops, results=dms_plot
        )
    else:
        dms_plot = get_meth_stats_for_granges_lib.DmrMethStatsDfs(
            mhpaths.project_temp_dir + "/asfasdfasdf"
        )


    dms_plot.cpg_betas

    beta_values_gr = pr.PyRanges(
        pd.concat(
            [
                dms_plot.cpg_annos[["Chromosome", "Start", "End"]],
                dms_plot.cpg_betas.rename(
                    columns=mhvars.ds1.nice_names_to_plot_names_d
                )[mhvars.ds1.plot_names_ordered],
            ],
            axis=1,
        )
    )

    plot_df = pd.melt(
        beta_values_gr["11", roi_start:roi_end].df,  # type: ignore
        id_vars=["Chromosome", "Start", "End"],
        var_name="subject",
        value_name="beta_value",
    )

    gencode_df = (
        gencode_gr["11", roi_start:roi_end]
        .df.rename(columns={"Feature": "feature"})
        .loc[lambda df: df["feature"].isin(["transcript", "exon", "UTR"])]
    )

    gene_ids_with_multiple_transcripts = (
        gencode_df.query('feature == "transcript"')
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

    # %%

    n_main_plots = plot_df.subject.nunique()
    anno_plots_abs_sizes = (cm(5),)
    figsize = (ut.cm(15), ut.cm(25))

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

    bar_plot(
        beta_values=plot_df,
        axes=axes[:n_main_plots],
        subject_order=mhvars.ds1.plot_names_ordered,
        # subject_order=["HSC", "Monocytes", "B cells"],
        region_boundaries=None,
        palette=mhstyle.hema_colors_ds1.plot_name_to_compartment_color_d,
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

    coutils.add_margin_label_via_encompassing_big_ax(
        fig=fig,
        axes=axes[:n_main_plots],
        big_ax=big_ax,
        axis="y",
        label="Methylation (%)",
    )

    plot_gene_model(
        df=gencode_df,
        offset=True,
        xlabel="Position (bp)",
        ax=axes[-1],
        roi=(roi_start, roi_end),
        rectangle_height=0.4,
        rectangle_height_utrs=0.2,
        perc_of_axis_between_arrows=0.03,
        arrow_length_perc_of_x_axis_size=0.01,
        arrow_height=0.1,
        gene_label_size=6,
    )

    ut.save_and_display(
        fig,
        png_path=mhpaths.project_temp_dir + "/asfsdf.png",
        # additional_formats=tuple(),
    )
    # %%
