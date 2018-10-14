import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional, List

from methlevels import MethStats
from methlevels.utils import NamedColumnsSlice as ncls

axes_context_despined = sns.axes_style('whitegrid', rc={
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.bottom': False,
    'axes.spines.top': False,
})


class DMRPlot:


    def __init__(self, meth_stats: MethStats,
                 region_id_col: str = 'region_id',
                 region_part_col: str = 'region_part') -> None:
        self.meth_stats = meth_stats
        subject_categories = list(self.meth_stats.df.columns
                                  .get_level_values('Subject').categories)
        self.subject_y = pd.Series(dict(zip(subject_categories,
                                            range(len(subject_categories)))))
        self.region_part_col = region_part_col
        self.region_id_col = region_id_col


    def __str__(self):
        return str({'df': str(self.meth_stats.df),
                    'anno': str(self.meth_stats.anno)})


    def grid_plot(self, output_dir, label, n_surrounding=5, draw_region_boundaries=True,
                  filetypes: Optional[List[str]] = None,
                  highlighted_subjects=None, row_height=0.4/2.54,
                  bp_width_100 = 1 / 2.54, bp_padding=100,
                  dpi=180) -> None:

        if filetypes is None:
            filetypes = ['png']

        if draw_region_boundaries:
            assert self.meth_stats.anno.columns.contains(self.region_part_col)

        for region_id, anno_group in self.meth_stats.anno.groupby(self.region_id_col):

            is_dmr = ~anno_group[self.region_part_col].isin(['left_flank', 'right_flank'])
            dmr_start = is_dmr.values.argmax()
            dmr_end = is_dmr.shape[0] - is_dmr.values[::-1].argmax() - 1
            dmr_slice = slice(max(dmr_start - n_surrounding, 0),
                              min(dmr_end + n_surrounding + 1, anno_group.shape[0]))
            anno_group = anno_group.iloc[dmr_slice]
            new_ms = self.meth_stats.update(anno=anno_group)
            tidy_df, tidy_anno = new_ms.to_tidy_format()
            title = f'{label}_{region_id}'

            figsize = (bp_width_100 * (tidy_df['Start'].iloc[-1] - tidy_df['Start'].iloc[0]) / 100,
                       tidy_df['Subject'].nunique() * row_height)

            with axes_context_despined, mpl.rc_context({'axes.grid.axis': 'y'}):
                ax: Axes
                fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=figsize, dpi=180)

                xlim = (tidy_df.iloc[0]['Start'] - bp_padding, tidy_df.iloc[-1]['Start'] + bp_padding)

                ax.set(xlim=xlim)
                ax.set_title(title)

                sc = ax.scatter(tidy_df['Start'] + 0.5, tidy_df['Subject'], c=tidy_df['beta_value'],
                                marker='.', edgecolor='black', linewidth=0.5, s=50,
                                cmap='YlOrBr', vmin=0, vmax=1,
                                zorder=10)

                if draw_region_boundaries:
                    curr_region_part = tidy_anno[self.region_part_col].iloc[0]
                    curr_start = tidy_anno['Start'].iloc[0]
                    last_boundary = 0
                    for unused_idx, curr_anno_row_ser in tidy_anno.iterrows():
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

                for suffix in filetypes:
                    fp = output_dir / f'{title}.{suffix}'
                    print('Saving', fp)
                    fig.savefig(fp)




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
