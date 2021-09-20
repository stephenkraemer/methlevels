from pathlib import Path
from io import StringIO
import matplotlib.pyplot as plt
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
from methlevels.plot_utils import cm
from methlevels.plot_genomic import (
    plot_gene_model,
    get_text_width_data_coordinates,
    plot_genomic_region_track,
)
from methlevels.plot_methlevels import bar_plot

import mouse_hema_meth.styling as mhstyle

mpl.rcParams.update(mhstyle.paper_context)
import mouse_hema_meth.utils as ut
import mouse_hema_meth.paths as mhpaths
import mouse_hema_meth.methylome.alignments_mcalls.get_meth_stats_for_granges_lib as get_meth_stats_for_granges_lib
import mouse_hema_meth.methylome.clustering.characterize_clustering_2_paths as characterize_clustering_2_paths
import mouse_hema_meth.shared_vars as mhvars
import matplotlib.gridspec as gridspec
import codaplot.utils as coutils


def test_plot_gene_model():

    gencode_df = pd.read_pickle(
        "/home/kraemers/projects/methlevels/tests/test-data/gencode_mm10_egf_region_bed-like-df.p"
    )

    fig, ax = plt.subplots(
        1, 1, dpi=180, figsize=(16 / 2.54, 3 / 2.54), constrained_layout=True
    )
    # bp_scale="Mb",
    # format_str="{:.3f}",
    plot_gene_model(
        df=gencode_df,
        offset=True,
        xlabel="Position (bp)",
        ax=ax,
        # roi=(roi_start, roi_end),
        rectangle_height=0.4,
        rectangle_height_utrs=0.2,
        perc_of_axis_between_arrows=0.03,
        arrow_length_perc_of_x_axis_size=0.01,
        arrow_height=0.1,
        gene_label_size=6,
    )
    ut.save_and_display(fig, png_path=mhpaths.project_temp_dir + "/asfsdf.png")


def test_barplot():

    # %%
    dms_all = get_meth_stats_for_granges_lib.dmr_meth_stats_all_pops_no_qc_poplevel

    plot_df = (
        dms_all.cpg_betas.rename(columns=mhvars.ds1.nice_names_to_plot_names_d)
        # [
        #     ["HSC", "Monocytes", "B cells"]
        # ]
        .loc[dms_all.cpg_annos.region_id.eq(100)]
        .stack()
        .reset_index()
        .set_axis(["cpg_id", "subject", "beta_value"], axis=1)
        .merge(dms_all.cpg_annos[["Chromosome", "Start", "End"]], on="cpg_id")[
            ["Chromosome", "Start", "End", "subject", "beta_value"]
        ]
    )

    # %%
    kwargs_to_test = dict(
        # a=dict(
        #     minimum_bar_width_pt=3,
        #     merge_overlapping_bars=False,
        #     barplot_lw=0.2,
        #     show_splines=True,
        # ),
        # b=dict(
        #     minimum_bar_width_pt=0,
        #     merge_overlapping_bars=False,
        #     barplot_lw=0.2,
        #     show_splines=True,
        # ),
        # c=dict(
        #     minimum_bar_width_pt=3,
        #     merge_overlapping_bars='bin',
        #     barplot_lw=0.2,
        #     show_splines=False,
        # ),
        # with_single_region_boundary_box=dict(
        #     minimum_bar_width_pt=3,
        #     merge_overlapping_bars='bin',
        #     barplot_lw=0,
        #     show_splines=True,
        #     region_properties=pd.DataFrame(
        #         dict(Chromosome=["1"], Start=np.unique(plot_df.Start)[1], End=np.unique(plot_df.End)[-2],
        #     )),
        #     region_boundaries="box",
        #     region_boundaries_kws = {"color": "gray", "alpha": 0.5},
        # ),
        # with_single_region_boundary_box_and_binned_bars=dict(
        #     minimum_bar_width_pt=3,
        #     merge_overlapping_bars='bin',
        #     barplot_lw=0,
        #     show_splines=True,
        #     region_properties=pd.DataFrame(
        #         dict(Chromosome=["1"], Start=np.unique(plot_df.Start)[1], End=np.unique(plot_df.End)[-3],
        #     )),
        #     region_boundaries="box",
        #     region_boundaries_kws = {"color": "gray", "alpha": 0.5},
        # ),
        # with_multiple_region_boundary_box_and_binned_bars=dict(
        #     minimum_bar_width_pt=3,
        #     merge_overlapping_bars='bin',
        #     barplot_lw=0,
        #     show_splines=True,
        #     region_properties=pd.DataFrame(
        #         dict(
        #             Chromosome=["1", "1"],
        #             Start=np.unique(plot_df.Start)[[0, 3]],
        #             End=np.unique(plot_df.End)[[1, -1]],
        #     )),
        #     region_boundaries="box",
        #     region_boundaries_kws = {"color": "gray", "alpha": 0.5},
        # ),
        with_single_region_boundary_box_and_dodged_bars=dict(
            minimum_bar_width_pt=3,
            merge_overlapping_bars='dodge',
            barplot_lw=0,
            show_splines=True,
            region_properties=pd.DataFrame(
                dict(Chromosome=["1"], Start=np.unique(plot_df.Start)[1], End=np.unique(plot_df.End)[-2],
            )),
            region_boundaries="box",
            region_boundaries_kws = {"color": "gray", "alpha": 0.5},
            min_gap_width_pt = 0.2,
        ),
        with_multiple_region_boundary_box_and_dodged_bars=dict(
            minimum_bar_width_pt=3,
            merge_overlapping_bars='dodge',
            barplot_lw=0,
            show_splines=True,
            region_properties=pd.DataFrame(
                dict(
                    Chromosome=["1", "1"],
                    Start=np.unique(plot_df.Start)[[0, 3]],
                    End=np.unique(plot_df.End)[[1, -1]],
            )),
            region_boundaries="box",
            region_boundaries_kws = {"color": "gray", "alpha": 0.5},
            min_gap_width_pt = 1,
        ),
    )

    for name, kwargs in kwargs_to_test.items():

        facet_grid_axes = coutils.FacetGridAxes(
            n_plots=plot_df.subject.nunique(),
            n_cols=1,
            figsize=(ut.cm(10), ut.cm(30)),
            constrained_layout_pads=dict(h_pad=0, w_pad=0.02, hspace=0, wspace=0),
            figure_kwargs=dict(constrained_layout=True, dpi=180),
        )

        bar_plot(
            beta_values=plot_df,
            axes=facet_grid_axes.axes_flat,
            subject_order=mhvars.ds1.plot_names_ordered,
            palette=mhstyle.hema_colors_ds1.plot_name_to_compartment_color_d,
            ylabel=None,
            axes_title_position="right",
            axes_title_size=6,
            axes_title_rotation=0,
            grid_lw=0.5,
            grid_color="lightgray",
            xlabel="Position (bp)",
            n_xticklabels=5,
            # n_yticklabels=3,
            ylim=(0, 1),
            yticks_major=(0, 1),
            yticks_minor=(0.5,),
            # xlim=(9857000, 9858000),
            # xticks=(9857000, 9857500, 9858000),
            **kwargs,
        )

        facet_grid_axes.add_y_marginlabel("Methylation (%)")

        ut.save_and_display(
            facet_grid_axes.fig,
            png_path=mhpaths.project_temp_dir + f"/asfsdf_{name}.svg",
            additional_formats=tuple(),
        )
    # %%


def test_plot_genomic_region_track():

    # %%
    fig, axes = plt.subplots(
        1, 3, constrained_layout=True, dpi=180, figsize=(cm(8), cm(3))
    )
    fig.set_constrained_layout_pads(hspace=0, wspace=0, h_pad=0, w_pad=0)

    granges_gr = pr.PyRanges(
        pd.read_csv(
            StringIO(
                """
Chromosome Start End name
1 100 200 name1
1 250 300 name2
1 380 400 name3
    """
            ),
            sep=" ",
        ).assign(  # type: ignore
            Start=lambda df: df.Start + 110_000_000, End=lambda df: df.End + 110_000_000
        )
    )

    # single color, no labels, no zoom, no explicit offset/order of magnitude
    plot_genomic_region_track(
        granges_gr=granges_gr,
        ax=axes[0],
        color="blue",
        show_names=False,
    )

    # palette, with labels, with zoom, no x axis
    plot_genomic_region_track(
        granges_gr=granges_gr,
        ax=axes[1],
        roi=(110_000_000 + 260, 110_000_000 + 410),
        palette={"name1": "red", "name2": "#ff3333", "name3": "black"},
        show_names=True,
        ax_abs_height=cm(3),
        label_size=6,
        no_coords=True,
    )

    # palette, with labels, with zoom, with x axis, adjusted y margin, offset
    plot_genomic_region_track(
        granges_gr=granges_gr,
        ax=axes[2],
        roi=(110_000_000 + 260, 110_000_000 + 410),
        palette={"name1": "red", "name2": "#ff3333", "name3": "black"},
        show_names=True,
        ax_abs_height=cm(3),
        label_size=6,
        no_coords=False,
        ymargin=0.1,
        offset=109_900_000,
    )

    fig
    # %%


# not up to date
# @pytest.fixture()
# def meth_calls_df():
#     meth_calls_df = read_csv_with_padding(
#         """
#         Subject    ,       ,     ,           , hsc    , hsc     , mpp4   , mpp4    , b-cells , b-cells
#         Stat  ,       ,     ,           , n_meth , n_total , n_meth , n_total , n_meth  , n_total
#         Chromosome , Start , End , region_id ,        ,         ,        ,         ,         ,
#         1          , 10    , 12  , 0         , 10      , 10      , 30      , 30      , 20      , 20
#         1          , 20    , 22  , 0         , 20      , 20      , 30      , 30      , 20      , 20
#         1          , 22    , 24  , 0         , 30      , 30      , 10      , 20      , 0      , 20
#         1          , 24    , 26  , 0         , 30      , 30      , 10      , 20      , 0      , 20
#         1          , 30    , 32  , 0         , 30      , 30      , 20      , 20      , 10      , 20
#         1          , 34    , 36  , 0         , 30      , 30      , 20      , 20      , 10      , 20
#         1          , 44    , 46  , 0         , 30      , 20      , 10      , 10      , 20      , 20
#         1          , 50    , 52  , 0         , 10      , 10      , 10      , 10      , 20      , 20
#     """,
#         header=[0, 1],
#         index_col=[0, 1, 2, 3],
#     )
#     return meth_calls_df


# @pytest.mark.parametrize("segmented", [False, True])
# @pytest.mark.parametrize("custom_region_part_col_name", [False, True])
# @pytest.mark.parametrize("multiple_regions", [False, True])
# def test_region_plot(
#     tmpdir, meth_calls_df, segmented, custom_region_part_col_name, multiple_regions
# ):

#     tmpdir = Path(tmpdir)

#     if custom_region_part_col_name:
#         region_id_col = "custom_region_id"
#         region_part_col = "custom_region_part"
#     else:
#         region_part_col = "region_part"
#         region_id_col = "region_id"

#     if segmented:
#         anno_df = pd.DataFrame(
#             {
#                 region_id_col: 0,
#                 region_part_col: np.repeat("left_flank dmr dms right_flank".split(), 2),
#             },
#             index=meth_calls_df.index,
#         )
#     else:
#         anno_df = pd.DataFrame(
#             {
#                 region_id_col: 0,
#                 region_part_col: np.repeat("left_flank dmr dmr right_flank".split(), 2),
#             },
#             index=meth_calls_df.index,
#         )

#     if multiple_regions:
#         anno_df2 = anno_df.copy()
#         anno_df2[region_id_col] = 1
#         anno_df2.rename(index={0: 1}, inplace=True)
#         anno_df = pd.concat([anno_df, anno_df2]).sort_index()
#         meth_calls_df2 = meth_calls_df.copy()
#         meth_calls_df2.rename({0: 1}, inplace=True)
#         meth_calls_df2.loc[:, ncls(Stat="n_meth")] = (
#             meth_calls_df2.loc[:, ncls(Stat="n_meth")] / 2
#         )
#         meth_calls_df = pd.concat([meth_calls_df, meth_calls_df2]).sort_index()

#     meth_stats = ml.MethStats(meth_calls_df, anno_df)

#     if custom_region_part_col_name:
#         region_plot = ml.DMRPlot(
#             meth_stats, region_part_col=region_part_col, region_id_col=region_id_col
#         )
#     else:
#         region_plot = ml.DMRPlot(meth_stats)

#     for highlighted_subjects in [None, ["mpp4"], ["hsc", "mpp4"]]:
#         region_plot.grid_plot(
#             tmpdir,
#             "mpp4-dmrs",
#             highlighted_subjects=highlighted_subjects,
#             bp_width_100=8,
#             row_height=1,
#             bp_padding=10,
#         )
