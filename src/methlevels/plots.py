from typing import Optional

from dpcontracts import invariant

import pandas as pd

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from dataclasses import dataclass
from pandas.util.testing import assert_frame_equal

from methlevels import MethStats, gr_names
from methlevels.methstats import _assert_tidy_meth_stats_data_contract, _assert_tidy_anno_contract
from methlevels.utils import NamedColumnsSlice as ncls

axes_context_despined = sns.axes_style('whitegrid', rc={
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.bottom': False,
    'axes.spines.top': False,
})




@invariant('meth_data format', lambda inst: _assert_tidy_meth_stats_data_contract(inst.meth_data))
@invariant('anno format', lambda inst: _assert_tidy_anno_contract(inst.anno, inst.meth_data))
class RegionPlot:

    def __init__(self, meth_data: pd.DataFrame, anno: Optional[pd.DataFrame],
                 region_part_col: str = 'region_part') -> None:
        self.meth_data = meth_data
        self.anno = anno
        subject_categories = list(self.meth_data.Subject.cat.categories)
        self.subject_y = pd.Series(dict(zip(subject_categories,
                                            range(len(subject_categories)))))
        self.region_part_col = region_part_col
        # assert list(meth_data.index.names[0:3]) == gr_names
        # assert meth_data.index.is_lexsorted()
        # self.meth_data = meth_data.stack(0).reset_index()

        # if anno is not None:
            # assert anno.index.names[0:3] == ['Chromosome', 'Start', 'End']
            # assert anno.index.is_lexsorted()
            # self.anno = (anno
            #              .drop(list(set(anno.index.names) & set(anno.columns)), axis=1)
            #              .reset_index())
        # else:
        #     self.anno = None


    def __str__(self):
        return str({'df': str(self.meth_data),
                    'anno': str(self.anno)})


    def grid_plot(self, title, draw_region_boundaries=True,
                  highlighted_subjects=None, row_height=0.4/2.54,
                  bp_width_100 = 1 / 2.54, bp_padding=100,
                  dpi=180):

        if draw_region_boundaries:
            assert self.anno.columns.contains(self.region_part_col)

        figsize = (bp_width_100 * (self.meth_data['Start'].iloc[-1] - self.meth_data['Start'].iloc[0]) / 100,
                   self.meth_data['Subject'].nunique() * row_height)

        with axes_context_despined, mpl.rc_context({'axes.grid.axis': 'y'}):
            fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize, dpi=180)
            ax: Axes

            xlim = (self.meth_data.iloc[0]['Start'] - bp_padding, self.meth_data.iloc[-1]['Start'] + bp_padding)

            ax.set(xlim=xlim)
            ax.set_title(title)

            sc = ax.scatter(self.meth_data['Start'] + 0.5, self.meth_data['Subject'], c=self.meth_data['beta_value'],
                            marker='.', edgecolor='black', linewidth=0.5, s=50,
                            cmap='YlOrBr', vmin=0, vmax=1,
                            zorder=10)

            if draw_region_boundaries:
                curr_region_part = self.anno[self.region_part_col].iloc[0]
                curr_start = self.anno['Start'].iloc[0]
                last_boundary = 0
                for unused_idx, curr_anno_row_ser in self.anno.iterrows():
                    if curr_anno_row_ser[self.region_part_col] != curr_region_part:
                        vline_x = curr_start + 0.5 + (curr_anno_row_ser['Start'] - curr_start) / 2
                        ax.axvline(vline_x, color='gray', linewidth=0.5, linestyle='--')
                        curr_region_part = curr_anno_row_ser[self.region_part_col]
                    curr_start = curr_anno_row_ser['Start']
                    # curr_start = ser['Start']


            if highlighted_subjects is not None:
                ax.barh(self.subject_y[highlighted_subjects], xlim[1], 0.8,
                        color='red', alpha=0.3)

            if fig is not None:
                fig.colorbar(sc, shrink=0.6)

        return fig



    def long_line_plot(self):

        df = self.meth_stats.df
        coord = df.index.to_frame()

        groupby_cols = ['Subject', 'Replicate'] if self.meth_stats.level == 'Replicate' else ['Subject']
        gp = df.groupby(level=groupby_cols, axis=1)


        #-
        n_samples = len(gp)
        height_line = 4/2.54
        height_bar = 3/2.54
        fig = plt.figure(figsize=(8/2.54, (height_line + height_bar) * n_samples), constrained_layout=True)
        gs = gridspec.GridSpec(n_samples * 2, 1,
                               height_ratios=[height_line, height_bar] * n_samples,
                               figure=fig,
                               hspace=0, wspace=0)

        line_plot_axes = []
        bar_axes = []

        for i, (name, df) in enumerate(gp):
            line_plot_ax = fig.add_subplot(gs[0 + i * 2, 0])
            line_plot_ax.plot(coord['Start'], df.loc[:, ncls(Stat='beta_value')].squeeze(), marker='.')
            line_plot_ax.set_title(name)
            line_plot_axes.append(line_plot_ax)
            bar_plot_ax = fig.add_subplot(gs[1 + (i - 1) * 2, 0])
            bar_plot_ax.bar(coord['Start'].values, df.loc[:, ncls(Stat='n_total')].squeeze(), color='gray')
            bar_axes.append(bar_plot_ax)

        plt.setp(line_plot_axes, ylim=(0, 1))
        plt.setp(bar_axes, ylim=(0, 30))

        #-
        return fig




# def region_plot(meth_stats: MethStats):
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     curr_plot_data = plot_data.loc[region_id, :].reset_index(drop=True)
#     sns.heatmap(curr_plot_data, ax=ax, cmap='YlOrBr')
#     fig.savefig(output_dir_region_cpg_meth / f'{region_id}_beta-values.png')
#
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     curr_plot_data_delta = curr_plot_data.subtract(curr_plot_data['hsc'], axis=0)
#     sns.heatmap(curr_plot_data_delta, ax=ax, cmap='RdBu_r', center=0)
#     fig.savefig(output_dir_region_cpg_meth / f'{region_id}_delta-meth.png')
#
# def plot_intervals(meth_in_regions_with_region_id_anno, output_dir_region_cpg_meth):
#     output_dir_region_cpg_meth.mkdir(exist_ok=True, parents=True)
#     pop_data = meth_in_regions_with_region_id_anno.convert_to_populations()
#
#     plot_data = pop_data.meth_stats.loc[:, idxs[:, 'beta_value']]
#     plot_data.columns = plot_data.columns.droplevel(1)
#     plot_data.index = pop_data.anno['region_id']
#     plot_data
#
#     for region_id, group_df in plot_data.groupby('region_id'):
#         print(region_id)
#
#         fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#         curr_plot_data = plot_data.loc[region_id, :].reset_index(drop=True)
#         sns.heatmap(curr_plot_data, ax=ax, cmap='YlOrBr')
#         fig.savefig(output_dir_region_cpg_meth / f'{region_id}_beta-values.png')
#
#         fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#         curr_plot_data_delta = curr_plot_data.subtract(curr_plot_data['hsc'], axis=0)
#         sns.heatmap(curr_plot_data_delta, ax=ax, cmap='RdBu_r', center=0)
#         fig.savefig(output_dir_region_cpg_meth / f'{region_id}_delta-meth.png')
#
#     plt.close('all')
#
# def plot_examples_for_pop_combi(bed_calls, df, segment_label, output_dir):
#     output_dir = Path(output_dir)
#     dmr_intervals = (df.query('segment_label == @segment_label')
#                      .sample(5, random_state=123).reset_index()[['chr', 'start', 'end']]
#                      .sort_values(['chr', 'start', 'end']))
#     cpg_meth = bed_calls.intersect(intervals_df=dmr_intervals, n_cores=24)
#     plot_intervals(meth_in_regions_with_region_id_anno = cpg_meth,
#                    output_dir_region_cpg_meth = output_dir / segment_label)
