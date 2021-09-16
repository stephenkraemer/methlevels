from pathlib import Path
from io import StringIO
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
from methlevels.region_plot import (
    restrict_to_primary_transcript_if_multiple_transcripts_are_present,
    region_plot,
)
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


def get_cpg_betas_gr():
    """Preparation for test_region_plot2, run once"""

    "tabix /omics/odcf/analysis/OE0219_projects/mouse_hematopoiesis/results/wgbs/results_per_pid/v1_bistro-0.2.0_odcf-alignment/hsc_1/meth/meth_calls/mcalls_hsc_1_CG_chrom-merged_strands-merged.bed.gz 11:18879817-19018985 | wc -l"

    # ===

    from mouse_hema_meth.methylome.alignments_mcalls.meth_calling_paths import (
        ds1_metadata_table_tsv,
    )

    ds1_metadata_table = pd.read_csv(ds1_metadata_table_tsv, sep="\t", header=0)

    roi_start = 19_018_985 - 1000
    roi_end = 19_018_985 + 5000

    hierarchy_bed_calls_ds1 = ml.BedCalls(
        metadata_table=ds1_metadata_table,
        tmpdir=mhpaths.project_temp_dir,
        pop_order=mhvars.ds1.all_pops,
        beta_value_col=6,
        n_meth_col=7,
        n_total_col=8,
    )

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

    beta_values = dms_plot.cpg_betas

    beta_values.to_csv(
        "/home/kraemers/projects/methlevels/tests/test-data/cpg-betas.tsv", sep="\t"
    )

    dms_plot.cpg_annos.to_csv(
        "/home/kraemers/projects/methlevels/tests/test-data/cpg-annos.tsv", sep="\t"
    )


def test_region_plot():

    cpg_betas = pd.read_csv(
        "/home/kraemers/projects/methlevels/tests/test-data/cpg-betas.tsv", sep="\t"
    )
    cpg_annos = pd.read_csv(
        "/home/kraemers/projects/methlevels/tests/test-data/cpg-annos.tsv", sep="\t"
    )

    beta_values_gr = pr.PyRanges(
        pd.concat(
            [
                cpg_annos[["Chromosome", "Start", "End"]],
                cpg_betas.rename(columns=mhvars.ds1.nice_names_to_plot_names_d)[
                    mhvars.ds1.plot_names_ordered
                ],
            ],
            axis=1,
        )
    )

    gtf_fp = (
        "/omics/odcf/analysis/OE0219_projects/mouse_hematopoiesis/databases/gene_annotations"
        "/gencode.vM19.annotation.no-prefix.gtf"
    )
    gencode_df = pr.read_gtf(gtf_fp, duplicate_attr=True, as_df=True)
    gencode_df = restrict_to_primary_transcript_if_multiple_transcripts_are_present(
        gencode_df
    )
    gencode_gr = pr.PyRanges(gencode_df)

    granges_gr_1 = pr.PyRanges(
        pd.read_csv(
            StringIO(
                """
Chromosome Start End name
11 100 300 name1
11 1250 1300 name2
11 2000 3000 name3
    """
            ),
            sep=" ",
        ).assign(
            Start=lambda df: df.Start + 19_018_985, End=lambda df: df.End + 19_018_985
        )
    )
    granges_gr_2 = pr.PyRanges(
        pd.read_csv(
            StringIO(
                """
Chromosome Start End name
11 100 300 NAME1
11 1250 1300 NAME2
11 2000 3000 NAME3
    """
            ),
            sep=" ",
        ).assign(
            Start=lambda df: df.Start + 19_018_985 + 200,
            End=lambda df: df.End + 19_018_985 + 200,
        )
    )

    fig, axes_d = region_plot(
        beta_values_gr=beta_values_gr,
        gencode_gr=gencode_gr,
        genomic_regions={"track1": granges_gr_1, "track2": granges_gr_2},
        plot_genomic_region_track_kwargs={
            "track1": {
                "palette": {"name1": "red"},
                "color": "blue",
                "show_names": True,
            },
            "track2": {"color": "red"},
        },
        chrom="11",
        start=19_018_985 - 1000,
        end=19_018_985 + 5000,
        offset=True,
        subject_order=mhvars.ds1.plot_names_ordered,
        palette=mhstyle.hema_colors_ds1.plot_name_to_compartment_color_d,
        anno_axes_size=cm(0.5),
        gene_axes_size=cm(1.5),
        anno_axes_padding=0.02,
        figsize=(cm(15), cm(20)),
        gene_model_kwargs=dict(
            rectangle_height=0.4,
            rectangle_height_utrs=0.2,
            perc_of_axis_between_arrows=0.03,
            arrow_length_perc_of_x_axis_size=0.01,
            arrow_height=0.1,
            gene_label_size=6,
            y_bottom_margin=-0.2,
        ),
        debug=False,
    )
    ut.save_and_display(
        fig,
        png_path=mhpaths.project_temp_dir + "/asfsdf.svg",
        # additional_formats=tuple(),
    )
    a = 3
