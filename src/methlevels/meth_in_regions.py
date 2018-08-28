"""Methylation in region stats handling

Can deal with non-consecutive repIDs in swarm plots
"""

#-

from functools import partial
from pathlib import Path
from typing import Optional

import statsmodels.stats.api as sms


import pandas as pd
import numpy as np
from pandas import IndexSlice as idxs
import re

"""python imports for seaborn plotting to file matplotlib"""
import matplotlib
from matplotlib.axes import Axes # for autocompletion in pycharm
from matplotlib.figure import Figure  # for autocompletion in pycharm
matplotlib.use('Agg') # import before pyplot import!
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as gg

from methlevels.utils import NamedColumnsSlice as ncls

class MethInRegions:
    def __init__(self, meth_stats: Optional[pd.DataFrame] = None, anno: Optional[pd.DataFrame] = None, filepath: Optional[str] = None,
                 pop_name_abbreviations=None, pop_order=None):
        self.pop_order = pop_order
        self.pop_name_abbreviations = pop_name_abbreviations
        self.anno = anno
        self.meth_stats = meth_stats
        self.filepath = filepath


    def read_meth_levels(self):
        """

        - chromosome name will be changed to chr, chrom and #chrom as input also allowed
        - uses pop order from df if none given, currently not save as attribute, only used in this function
        - names: beta, n_meth, n_total; also recognizes beta_value, renames to beta
        """

        if self.filepath.endswith('.feather'):
            meth_stats_with_anno = pd.read_feather(self.filepath)
        else:
            meth_stats_with_anno = pd.read_pickle(self.filepath)

        grange_col_name_mapping = {
            'chrom': 'chr',
            '#chrom': 'chr',
            '#chr': 'chr',
            'Chromosome': 'chr',
            'Start': 'start',
            'End': 'end',
        }
        meth_stats_with_anno = meth_stats_with_anno.rename(columns = chrom_name_mapping)

        # Not sure whether it is a feather/R bug that start and end are float
        # or whether this is something I have to fix on my side
        meth_stats_with_anno[['start', 'end']] = (
            meth_stats_with_anno[['start', 'end']].astype(int))
        # not sure whether conversion to string categorical is necessary
        if meth_stats_with_anno['chr'].dtype.name != 'category':
            meth_stats_with_anno[['chr']] = (meth_stats_with_anno['chr']
                                             .astype(str).astype('category'))
        meth_stats_with_anno = (meth_stats_with_anno
                                .set_index(['chr', 'start', 'end']))


        # Convert columns str index to MultiIndex
        # flat index is necessary for compatibility with feather/R
        # df has index columns and meth stat columns of the form
        # {pop}_{repID}_{stat_name}, e.g. hsc_1_beta. repID may be all for pop-level
        # stats, e.g. hsc_all_n_meth
        meth_stat_regex = (rf"([\w-]+)"
                           rf"_(\d+|all)"
                           f"_(beta_value|n_meth|n_total)")
        match_objects = [re.match(meth_stat_regex, s)
                         for s in meth_stats_with_anno.columns]

        columns_idx_tuples = []
        anno_cols = []
        for orig_col_name, mobj in zip(
                meth_stats_with_anno.columns, match_objects):
            if mobj is None:
                columns_idx_tuples.append((orig_col_name, 'None', 'None'))
                anno_cols.append(orig_col_name)
            else:
                columns_idx_tuples.append(mobj.groups())

        # multi_idx_arrays = list(zip(*columns_idx_tuples))
        # pop_names_in_order_of_appearance = multi_idx_arrays[0]

        # if self.pop_order:
        #     categories = self.pop_order
        # else:
        #     categories = pop_names_in_order_of_appearance
        #
        # multi_idx_arrays[0] = pd.Categorical(
        #     pop_names_in_order_of_appearance,
        #     categories=categories, ordered=True)
        #
        # multi_idx_arrays[1] = pd.Categorical(
        #     multi_idx_arrays[1],
        #     categories=pd.Series(
        #         multi_idx_arrays[1]).astype(int).sort_values().astype(str),
        #     ordered=True)
        multi_idx = pd.MultiIndex.from_tuples(columns_idx_tuples)
        meth_stats_with_anno.columns = multi_idx
        # sorting for indexing later on
        meth_stats_with_anno = meth_stats_with_anno.sort_index(axis=1)

        if anno_cols:
            meth_stats = meth_stats_with_anno.drop(anno_cols, axis=1)
            region_anno = meth_stats_with_anno.loc[:, anno_cols]
            region_anno.columns = region_anno.columns.droplevel([1,2])
        else:
            meth_stats = meth_stats_with_anno
            region_anno = None

        assert not meth_stats.isnull().all().all(), 'All meth stats are NA'

        meth_stats = meth_stats.rename(columns=self.pop_name_abbreviations, level=0)

        if self.pop_order:
            pop_order = (pd.Series(self.pop_order)
                         .replace(self.pop_name_abbreviations).tolist())
        else:
            pop_order = meth_stats.columns.get_level_values(0).unique()

        meth_stats_pops = meth_stats.columns.get_level_values(0).unique()
        if not (set(meth_stats_pops) == set(pop_order)):
            raise ValueError('Populations in meth stats and in pop order'
                             'vector dont match.'
                             f'Meth stats: {meth_stats_pops}'
                             f'Pop order: {pop_order}')

        pop_categorical_idx = pd.Categorical(
            meth_stats.columns.get_level_values(0),
            categories=pop_order, ordered=True)

        rep_idx_values = meth_stats.columns.get_level_values(1).to_series()
        all_reps_str_replacement_value = rep_idx_values.loc[rep_idx_values != 'all'].astype(int).max() + 1
        dummy_rep_idx_values = rep_idx_values.replace({'all': all_reps_str_replacement_value})
        # noinspection PyUnresolvedReferences
        rep_categories = (pd.Series(dummy_rep_idx_values.unique())
                          .astype(int)
                          .sort_values()
                          .astype(str)
                          .replace({str(all_reps_str_replacement_value): 'all'})
                          )
        rep_categorical_idx =  pd.Categorical(rep_idx_values,
                                              categories=rep_categories.values,
                                              ordered=True)

        meth_stats.columns = pd.MultiIndex.from_arrays(
            [pop_categorical_idx,
             rep_categorical_idx,
             meth_stats.columns.get_level_values(2)])
        meth_stats = meth_stats.sort_index(axis=1)

        meth_stats.columns.names = ['pop', 'rep', 'stat']

        self.meth_stats = meth_stats
        self.anno = region_anno

        return self


    def apply_coverage_limit(self, threshold):
        # noinspection PyUnresolvedReferences
        has_enough_coverage = (self.meth_stats
                               .loc[:, idxs[:, :, 'n_total']] >= threshold).all(axis=1)
        new_meth_stats = self.meth_stats.loc[has_enough_coverage, :].copy()
        if self.anno is not None:
            new_anno = self.anno.loc[has_enough_coverage, :].copy()
        else:
            new_anno = None
        return MethInRegions(meth_stats=new_meth_stats, anno=new_anno)

    def apply_fraction_loss_capped_coverage_limit(
            self, coverage_threshold, max_fraction_lost):


        coverage_stats = self.meth_stats.loc[:, ncls(stat='n_total')]
        per_sample_quantiles = coverage_stats.quantile(max_fraction_lost, axis=0)
        per_sample_thresholds = (per_sample_quantiles
                                 .where(per_sample_quantiles < coverage_threshold,
                                        coverage_threshold))
        coverage_mask: pd.DataFrame = coverage_stats >= per_sample_thresholds
        coverage_ok_in_all_samples = coverage_mask.all(axis=1)
        new_meth_stats = self.meth_stats.loc[coverage_ok_in_all_samples, :].copy()
        if self.anno is None:
            new_anno = None
        else:
            new_anno = self.anno.loc[coverage_ok_in_all_samples, :].copy()
        return MethInRegions(
            meth_stats=new_meth_stats,
            anno=new_anno
        )

    def apply_size_limit(self, n_cpg_threshold):
        is_large_enough = self.anno.loc[:, 'n_cpg'] >= n_cpg_threshold
        new_meth_stats = self.meth_stats.loc[is_large_enough, :].copy()
        if self.anno is None:
            new_anno = None
        else:
            new_anno = self.anno.loc[is_large_enough, :].copy()
        return MethInRegions(meth_stats=new_meth_stats, anno=new_anno)

    def restrict_to_replicates(self):
        rep_meth_stats = self.meth_stats.drop('all', level=1, axis=1).copy()
        # noinspection PyUnresolvedReferences
        rep_levels_only_idx = (rep_meth_stats
                               .columns
                               .get_level_values(1)
                               .remove_unused_categories())
        new_multiidx = pd.MultiIndex.from_arrays([
            rep_meth_stats.columns.get_level_values(0),
            rep_levels_only_idx,
            rep_meth_stats.columns.get_level_values(2),
        ])
        rep_meth_stats.columns = new_multiidx
        if self.anno is None:
            anno_copy = None
        else:
            anno_copy = self.anno.copy()
        return MethInRegions(meth_stats=rep_meth_stats, anno=anno_copy)

    def restrict_to_populations(self):
        pop_meth_stats = self.meth_stats.loc[:, idxs[:, 'all']].copy()
        if self.anno is None:
            anno_copy = None
        else:
            anno_copy = self.anno.copy()
        return MethInRegions(meth_stats=pop_meth_stats, anno=anno_copy)

    def convert_to_populations(self):
        pop_stats_df = self.meth_stats.groupby(level=['pop', 'stat'], axis=1).sum()

        def update_beta_values(group_df):
            group_df[group_df.name, 'beta_value'] = group_df[group_df.name, 'n_meth'] / group_df[group_df.name, 'n_total']
            return group_df
        pop_stats_df = pop_stats_df.groupby(level='pop', axis=1).apply(update_beta_values)

        if self.anno is None:
            anno_copy = None
        else:
            anno_copy = self.anno.copy()

        return MethInRegions(meth_stats=pop_stats_df, anno=anno_copy)


    def restrict_to_pop_or_rep_level(self, level):
        if level == 'pop':
            return self.restrict_to_populations()
        elif level == 'rep':
            return self.restrict_to_replicates()
        else:
            ValueError(f'Unknown level {level}')

    def plot_heatmap(self, n_features):
        """This needs to be """
        heatmap_data = self.meth_stats.sample(n_features)

    def sample(self, n, random_state=123):
        if 0 < n < self.meth_stats.shape[0]:
            new_meth_stats = self.meth_stats.sample(n, random_state=random_state).sort_index()
            new_anno = self.anno.loc[new_meth_stats.index, :]
            return MethInRegions(meth_stats=new_meth_stats, anno=new_anno)
        else:
            return self

    def get_most_var_betas(self, n):
        if 0 < n < self.meth_stats.shape[0]:
            most_var_idx = (self.meth_stats
                            .loc[:, ncls(stat='beta_value')]
                            .var(axis=1)
                            .sort_values(ascending=False)
                            .iloc[0:n]
                            .index
                            )
            new_meth_stats = self.meth_stats.loc[most_var_idx, :].sort_index()
            new_anno = self.anno.loc[new_meth_stats.index, :]
            return MethInRegions(meth_stats=new_meth_stats, anno=new_anno)
        else:
            return self

    def choose_n_observations(self, n, method='sample', random_state=123):
        if method=='sample':
            return self.sample(n, random_state)
        elif method=='mostvar':
            return self.get_most_var_betas(n)

    def __str__(self):
        str(self.meth_stats.head())

    def drop_regions(self, bool_idx):
        new_meth_stats = self.meth_stats.loc[bool_idx, :]
        new_anno = self.anno.loc[bool_idx, :]
        return MethInRegions(new_meth_stats, new_anno)

    def qc_filter(self, coverage=None, size=None, min_delta=None):

        meth_in_regions = self
        if coverage:
            meth_in_regions = meth_in_regions.apply_fraction_loss_capped_coverage_limit(
                coverage_threshold=coverage,
                max_fraction_lost=0.2
            )

        if size:
            meth_in_regions = meth_in_regions.apply_size_limit(size)

        if min_delta:

            betas = meth_in_regions.meth_stats.loc[:, ncls(stat='beta_value')]

            abs_deltas = betas.max(axis=1) - betas.min(axis=1)
            has_sufficient_delta = abs_deltas > min_delta

            meth_in_regions = MethInRegions(
                meth_stats=meth_in_regions.meth_stats.loc[has_sufficient_delta, :],
                anno=meth_in_regions.anno.loc[has_sufficient_delta, :],
            )

        return meth_in_regions

    def get_flat_columns(self):
        return ['_'.join(x) for x in self.meth_stats.columns]

    # old interface, discard when I am sure this is not used anymore
    def to_tsv(self, fp):
        raise NotImplemented
    def to_feather(self, fp):
        raise NotImplemented
    def to_df_pickle(self, fp):
        raise NotImplemented

    def save(self, filepaths, pop_name_abbreviations=None):
        multi_idx = self.meth_stats.columns
        if pop_name_abbreviations:
            reverse_abbreviation_mapping = {v:k for k,v in pop_name_abbreviations.items()}
            self.meth_stats = self.meth_stats.rename(reverse_abbreviation_mapping,
                                                     axis=1, level=0)
        self.meth_stats.columns = self.get_flat_columns()
        res = pd.concat([self.anno, self.meth_stats], axis=1).reset_index()
        for fp in filepaths:
            if fp.endswith('.feather'):
                res.to_feather(fp)
            elif fp.endswith('.tsv'):
                res.to_csv(fp, sep='\t', header=True, index=False)
            elif fp.endswith('.p'):
                res.to_pickle(fp)
            else:
                ValueError(f'Unknown suffix for {fp}, abort saving')
        self.meth_stats.columns = multi_idx


#-

def snakemake_filter_meth_in_regions(input, output, wildcards, params):
    qc_filter_params = dict(
        coverage = int(wildcards.coverage),
        size = int(wildcards.size),
        min_delta = float(wildcards.delta),
        level = wildcards.level,
    )
    choose_n_params = dict(
        n = int(wildcards.n),
        method = wildcards.samplingMethod,
        level = wildcards.level,
        random_state = 123,
    )
    meth_in_regions = (MethInRegions(filepath=input.full_meth_levels,
                                     pop_name_abbreviations=params.pop_name_abbreviations,
                                     pop_order=params.pop_order)
                       .read_meth_levels()
                       .qc_filter(**qc_filter_params)
                       .choose_n_observations(**choose_n_params))
    meth_in_regions.save([output.p, output.feather, output.tsv],
                         params.pop_name_abbreviations)


def corr_heatmap(meth_in_regions: MethInRegions, filepath,
                 pop_name_abbreviations,
                 samples, height_cm, width_cm):

    samples = pd.Series(samples).replace(pop_name_abbreviations).tolist()

    height_in = height_cm / 2.54
    width_in = width_cm / 2.54

    if not samples:
        samples = slice(None)
    meth_in_regions = meth_in_regions.restrict_to_replicates()
    meth_in_regions.meth_stats.head()
    beta_values = (meth_in_regions.meth_stats
                       .loc[:, (samples, slice(None), 'beta_value')])
    beta_values.columns = beta_values.columns.droplevel(2)
    beta_values.columns = ['_'.join(level_elems) for level_elems in zip(beta_values.columns.get_level_values(0),
                                                                        beta_values.columns.get_level_values(1))]
    beta_values.head()
    corr_mat = beta_values.corr()
    corr_mat
    # ro.globalenv['corr_mat'] = corr_mat

    # filepaths_by_ext = {'png': filepath,
    #                     'pdf': filepath.replace('png', 'pdf'),
    #                     'svg': filepath.replace('png', 'svg')}

    matplotlib.rcParams.update({'font.size': 5})
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    sns.heatmap(corr_mat,
                vmin=0, vmax=1,
                cmap='RdBu_r',
                fmt='.2g',
                annot=True,
                xticklabels=True,
                yticklabels=True,
                linewidths=0.15,
                linecolor='white',
                ax=ax)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)
    fig.set_size_inches(h=height_in, w=width_in)
    save_mpl_fig_with_different_filetypes_same_folder(
        fig=fig,
        fp=filepath,
        extensions=['png', 'pdf', 'svg'],
        dpi=120)


    # if add_values:
    #     cell_fun_code = '''\
    #     cell_fun = function(j, i, x, y, width, height, fill) {
    #         grid.text(sprintf("%.2f", corr_mat[i, j]), x, y, gp = gpar(fontsize = 5))
    #     }'''
    # else:
    #     cell_fun_code = ''
    #
    # ro.r(f"""
    #       library(ComplexHeatmap)
    #       library(circlize)
    #       print('Creating heatmap')
    #       h = Heatmap(corr_mat,
    #           cluster_columns = T, show_column_dend = T,
    #           cluster_rows = T, show_row_dend = T,
    #           show_row_names = T,
    #           show_column_names = TRUE, column_names_side = "top",
    #           col=colorRamp2(c(0,0.5,1), c("dark green", "yellow", "red")),
    #           heatmap_legend_param = list(title = "Corr.", color_bar = "continuous"),
    #           {cell_fun_code}
    #       )
    #       print('Saving as pdf, png, svg')
    #       pdf('{filepaths_by_ext["pdf"]}', height={height_in}, width={width_in})
    #       draw(h)
    #       dev.off()
    #       png('{filepaths_by_ext["png"]}', height={height_in}, width={width_in}, units='in', res=100)
    #       draw(h)
    #       dev.off()
    #       svg('{filepaths_by_ext["svg"]}', height={height_in}, width={width_in})
    #       draw(h)
    #       dev.off()
    # """)

#'

def test_single_region_meth_dist_plots():
    # meth_levels_fp = '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/meth_in_regions/dmrs/d3a-topdown/pw-segm/all_7-11-2017/mct/d3a_meth-levels.feather'
    # meth_levels_fp = '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/meth_in_regions/genome-tiles/500bp-windows/d3a/methylctools/500bp-windows_d3a_meth-levels.feather'
    # meth_levels_fp = '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/temp/line_plot_sandbox/results/pub-hsc-dcs_all-7-11-2017_mct/ensembl_regulatory_regions/multi-cell/feature-enhancer_status-undef/meth-levels.feather'
    meth_levels_fp = '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/temp/meth_in_regions_sandbox_feb26/results/d3a_all-7-11-2017_mct/dmr-masked/d3a-topdown_pw-segm/all-7-11-2017_mct/repeats/repeats_by_family/class-ltr/family-erv1/meth-levels.feather'

    output_dir = Path('/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/temp/feb28/wtf-repeats/')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {k: str(output_dir / v)
                    for k, v in {
                        'swarm': 'swarm.png',
                        'swarm_errorbars': 'swarm-errbar.png',
                        'swarm_trendline': 'swarm-trendline.png',
                        'violin_pop': 'violin_pop.png',
                        'violin_rep': 'violin_rep.png',
                        'box_pop': 'box_pop.png',
                        'box_rep': 'box_rep.png',
                    }.items()
                    }

    pop_name_abbreviations = {
            "lsk_mu-d3a-r882h_tr-aza": "KI/AZA",
            "lsk_mu-d3a-r882h_tr-none": "KI",
            "lsk_mu-d3a-wt_tr-aza": "AZA",
            "lsk_mu-d3a-wt_tr-none": "WT",
        }
    pop_order = ['WT', 'KI', 'AZA', 'KI/AZA']
    # pop_order = [
    #     'hsc', 'mpp1', 'mpp2', 'mpp34',
    #     'cdp', 'mdp', 'cmop',
    #     'pdc', 'dc-cd11b', 'dc-cd8a',
    # ]

    # # tested with pub-hsc-dcs
    # swarm_plot_params = {'offset': 0.15,
    #                      'height': 4,
    #                      'width': 4,
    #                      'rep_point_size': 0.4,
    #                      'mean_point_size': 0.4}

    # tested with d3a
    swarm_plot_params = {'offset': 0.15,
                         'height': 3,
                         'width': 3,
                         'rep_point_size': 0.4,
                         'mean_point_size': 0.4}

    n_elements_violin_plot = 10_000

    # d3a
    violin_plot_params = {
        'height': 4,
        'pop_plot_width': 3,
        'rep_plot_width': 5,
    }

    create_single_region_meth_dist_plots(
        meth_levels_fp=meth_levels_fp,
        output_paths=output_paths,
        pop_name_abbreviations=pop_name_abbreviations,
        pop_order=pop_order,
        n_elements_violin_plot = n_elements_violin_plot,
        swarm_plot_params=swarm_plot_params,
        violin_plot_params=violin_plot_params
    )

def create_single_region_meth_dist_plots(
        meth_levels_fp, output_paths, swarm_plot_params, violin_plot_params,
        pop_name_abbreviations, pop_order, n_elements_violin_plot,
        coverage_threshold_rep, coverage_threshold_pop,
        max_fraction_lost_due_to_coverage_filtering):

    base_args = dict(
        meth_levels_fp=meth_levels_fp,
        output_paths=output_paths,
        pop_name_abbreviations=pop_name_abbreviations,
        pop_order=pop_order,
        coverage_threshold_rep = coverage_threshold_rep,
        max_fraction_lost_due_to_coverage_filtering = max_fraction_lost_due_to_coverage_filtering,
    )

    create_average_distribution_plots(**base_args,
                                      swarm_plot_params=swarm_plot_params)

    create_violin_plots(n_elements=n_elements_violin_plot, **base_args,
                        coverage_threshold_pop = coverage_threshold_pop,
                        violin_plot_params=violin_plot_params)


def create_average_distribution_plots(meth_levels_fp,
                                      output_paths, pop_name_abbreviations,
                                      swarm_plot_params: dict,
                                      coverage_threshold_rep,
                                      max_fraction_lost_due_to_coverage_filtering,
                                      pop_order):

    print('Creating average meth level plots')

    swarm_plot_params = pd.Series(swarm_plot_params)

    meth_in_regions = MethInRegions(filepath=meth_levels_fp,
                                    pop_name_abbreviations=pop_name_abbreviations,
                                    pop_order=pop_order)
    meth_in_regions.read_meth_levels()

    coverage_filter_args = (
        coverage_threshold_rep, max_fraction_lost_due_to_coverage_filtering)
    rep_level_stats = (meth_in_regions
          .restrict_to_replicates()
          .apply_fraction_loss_capped_coverage_limit(*coverage_filter_args)
          .meth_stats.loc[:, idxs[:, :, 'beta_value']]
          .mean()
          .to_frame('mean_beta')
          .reset_index()
          .drop('stat', axis=1)
          )

    pops = rep_level_stats['pop'].cat.categories.tolist()
    base_x_mapping = dict(list(zip(pops, range(1, len(pops) + 1))))
    n_reps_per_pop = rep_level_stats.groupby('pop').size()

    def add_consecutive_indices(group_df):
        n_reps = group_df.shape[0]
        max_rep_id = int(group_df['rep'].max())
        if max_rep_id > n_reps:
            group_df = group_df.sort_values('rep')
            group_df['consecutive_rep_id'] = np.arange(1, n_reps+1)
        else:
            group_df['consecutive_rep_id'] = group_df['rep']
        return group_df
    rep_level_stats = rep_level_stats.groupby('pop').apply(add_consecutive_indices)

    def get_dodge_offsets(point_idx, n_points_total, offset_width):
        if n_points_total % 2 == 0:
            offsets = np.concatenate([np.arange(-n_points_total/2, 0),
                                      np.arange(1, (n_points_total/2) + 1)])
        else:
            x = (n_points_total-1)/2
            offsets = np.arange(-x, x+1)
        return offsets[point_idx] * offset_width
    rep_level_stats['x'] = rep_level_stats.apply(
        lambda ser: base_x_mapping[ser['pop']]
                    + get_dodge_offsets(
            int(ser['consecutive_rep_id']) - 1, n_reps_per_pop[ser['pop']],
            swarm_plot_params['offset']), axis=1)

    pop_level_stats = (rep_level_stats
                       .groupby('pop')['mean_beta']
                       .agg(['mean', 'sem'])
                       .reset_index()
                       .rename(columns = {'mean': 'mean_beta'}))
    pop_level_stats['x'] = pop_level_stats['pop'].replace(base_x_mapping)
    confidence_interval_boundaries = (rep_level_stats
                                      .groupby('pop')
                                      .apply(lambda df: pd.Series(sms.DescrStatsW(df['mean_beta']).tconfint_mean(alpha=0.1), index=['lower', 'upper'])))
    pop_level_stats = pd.concat([pop_level_stats.set_index(['pop']), confidence_interval_boundaries], axis=1).reset_index()
    # double check: ok
    # pop_level_stats['low'] = pop_level_stats.eval('mean_beta - 4 * sem')
    # pop_level_stats['upp'] = pop_level_stats.eval('mean_beta + 4 * sem')

    plot_fn = partial(average_methylation_plot,
                      pop_level_stats=pop_level_stats,
                      rep_level_stats=rep_level_stats,
                      width=swarm_plot_params['width'], height=swarm_plot_params['height'],
                      **swarm_plot_params[['rep_point_size', 'mean_point_size']])
    plot_fn(errorbars = True, trendline = True,
            out_png = output_paths['swarm_trendline'])
    plot_fn(errorbars = True, trendline = False,
            out_png = output_paths['swarm_errorbars'])
    plot_fn(errorbars = False, trendline = False,
            out_png = output_paths['swarm'])


# average_methylation_plot = ro.r('''
#     average_methylation_plot = function(pop_level_stats, rep_level_stats,
#                                         errorbars, trendline,
#                                         out_png, width, height,
#                                         rep_point_size,
#                                         mean_point_size) {
#       g = ggplot(pop_level_stats, aes(x = x, y = mean_beta))
#       if (trendline) {
#         g = g + geom_line(size=0.1)
#       }
#       # why do i use position_dodge - unnecessary now that I have my own
#       # offset computation?
#       g = g + geom_point(data = rep_level_stats,
#                          mapping = aes(color=rep),
#                          size=rep_point_size, shape=16, alpha=0.8)
#       if (errorbars) {
#         g = g + geom_errorbar(mapping=aes(ymin=lower, ymax=upper),
#                               width=0.2, size=0.1)
#       }
#       if (errorbars) {
#         g = g + geom_point(color='black', shape=18, size=mean_point_size, stroke=0.2)
#       }
#       g = g + scale_y_continuous(limits = c(0,1)) +
#         scale_x_continuous(
#           breaks = seq(1, length(levels(pop_level_stats[['pop']]))),
#           labels = levels(pop_level_stats[['pop']])) +
#         theme_classic(base_size=6) +
#         labs(y='Average methylation', color='Repl.') +
#         theme(axis.title.x = element_blank()) +
#         theme(legend.key.height = unit(0.3, 'cm'),
#               legend.key.width = unit(0.2, 'cm'),
#               legend.margin=margin(0,0,0,0),
#               legend.box.margin=margin(0,0,0,0),
#               legend.box.spacing=unit(0.2, 'cm'),
#               plot.margin=unit(c(0.1,0.1,0.1,0.1), 'cm'))
#       # if (length(levels(pop_level_stats[['pop']])) > 4) {
#       if (TRUE) {
#         g = g + theme(axis.text.x = element_text(angle=90, hjust=1, vjust=0.5))
#       }
#       ggsave(out_png, width=width, height=height, units='cm')
#       ggsave(str_replace(out_png, 'png', 'pdf'),
#              width=width, height=height, units='cm')
#       cat('saved to ', out_png)
#     }
# ''')

def create_violin_plots(meth_levels_fp, n_elements,
                        pop_name_abbreviations, pop_order,
                        coverage_threshold_pop,
                        coverage_threshold_rep,
                        max_fraction_lost_due_to_coverage_filtering,
                        violin_plot_params: dict,
                        output_paths):

    print('Creating violin plots')

    violin_plot_params = pd.Series(violin_plot_params)

    coverage_threshold_mapping = {'pop': coverage_threshold_pop,
                                  'rep': coverage_threshold_rep}

    meth_in_regions = MethInRegions(filepath=meth_levels_fp,
                                    pop_name_abbreviations=pop_name_abbreviations,
                                    pop_order=pop_order)
    meth_in_regions.read_meth_levels()

    for level_name in ['rep', 'pop']:

        print(level_name)

        drop_list = ['chr', 'start', 'end', 'stat']
        if level_name == 'pop':
            drop_list += ['rep']


        plot_data = (
            meth_in_regions
                .restrict_to_pop_or_rep_level(level_name)
                .apply_fraction_loss_capped_coverage_limit(
                    coverage_threshold=coverage_threshold_mapping[level_name],
                    max_fraction_lost=max_fraction_lost_due_to_coverage_filtering)
                .meth_stats
                .loc[:, idxs[:, :, 'beta_value']]
        )

        if 0 < n_elements < plot_data.shape[0]:
            plot_data = plot_data.sample(n_elements)

        plot_data = (plot_data.stack([0,1,2])
                     .to_frame('beta_value')
                     .reset_index()
                     .drop(drop_list, axis=1)
                     )

        plot_width_key = f'{level_name}_plot_width'
        violin_plot(plot_data, level_name,
                    height=violin_plot_params['height'],
                    width=violin_plot_params[plot_width_key],
                    out_png=output_paths[f'violin_{level_name}'])

# violin_plot = ro.r('''
#     violin_plot = function(plot_data, level_name, height, width, out_png) {
#       if (level_name == 'pop') {
#         mapping = aes(x = pop, y = beta)
#         width = 4
#       } else {
#         mapping = aes(x = pop, y = beta, fill=rep)
#         width = 8
#       }
#       g = ggplot(plot_data, mapping)
#       if (level_name == 'pop') {
#         g = g + geom_violin(draw_quantiles=c(0.25, 0.5, 0.75),
#                             adjust=1.5, trim=TRUE, fill='gray90',
#                             size=0.2)
#       } else {
#         g = g + geom_violin(draw_quantiles=c(0.25, 0.5, 0.75),
#                             adjust=1.5, trim=TRUE, size=0.2)
#       }
#       g = g + labs(y='Average methylation') +
#               theme_classic(6) +
#               theme(axis.title.x = element_blank(),
#                     legend.key.height = unit(0.2, 'cm'),
#                     legend.key.width = unit(0.2, 'cm'),
#                     legend.margin=margin(0,0,0,0),
#                     legend.box.margin=margin(0,0,0,0),
#                     legend.box.spacing=unit(0.2, 'cm'),
#                     plot.margin=unit(c(0.1,0.1,0.1,0.1), 'cm'))
#       if (level_name == 'rep') {
#         g = g + labs(fill = 'RepID')
#       }
#       if (length(levels(plot_data[['pop']])) > 4) {
#         g = g + theme(axis.text.x = element_text(angle=90, hjust=1, vjust=0.5))
#       }
#       ggsave(out_png, height=height, width=width, units='cm')
#       ggsave(str_replace(out_png, 'png', 'pdf'), height=height, width=width, units='cm')
#     }
# ''')


# From correlation plot
# =============================

# meth_levels_fp = '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/meth_in_regions/dmrs/d3a-topdown/pw-segm/all_7-11-2017/mct/d3a_meth-levels.feather'
# # meth_levels_fp = '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/meth_in_regions/genome-tiles/500bp-windows/d3a/methylctools/500bp-windows_d3a_meth-levels.feather'
# output_dir = '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/temp/feb15'
# pattern = 'd3a-segments_corr-heatmap_mincov-{mincov}_minsize-{minsize}_show-values-{show_values}_level-{level}_samples-{sample_name}.png'
# pattern = op.join(output_dir, pattern)
#
# pop_name_abbreviations = {
#     "lsk_mu-d3a-r882h_tr-aza": "KI/AZA",
#     "lsk_mu-d3a-r882h_tr-none": "KI",
#     "lsk_mu-d3a-wt_tr-aza": "Aza",
#     "lsk_mu-d3a-wt_tr-none": "WT",
# }
#
# # corr_heatmap_fp = '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/temp/feb15/all-samples_heatmap.png'
# # corr_heatmap_fp_no_treat = '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/temp/feb15/ki-wt-samples_heatmap.png'
# # corr_heatmap_fp_nofilter = '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/temp/feb15/all-samples_heatmap_no-filter.png'
# # corr_heatmap_fp_no_treat_nofilter = '/icgc/dkfzlsdf/analysis/hs_ontogeny/results/wgbs/cohort_results/temp/feb15/ki-wt-samples_heatmap_no-filter.png'
# #
#
# meth_in_regions = MethInRegions(filepath=meth_levels_fp,
#                                 pop_name_abbreviations=pop_name_abbreviations)
# meth_in_regions_by_level = {'pop': meth_in_regions.restrict_to_populations(),
#                             'rep': meth_in_regions.restrict_to_replicates()}
# corr_heatmap_with_size_partial = partial(corr_heatmap, height_cm=10, width_cm=10)
# sample_definitions = {'kiwt': ['KI', 'WT']}
# for (mincov, minsize), show_values, level, sample_name in itertools.product(
#         ((0,0), (10, 3)), (True, False), ('rep', 'pop'), ('kiwt', 'all')):
#     data = meth_in_regions_by_level[level]
#     if sample_name == 'all':
#         samples = None
#     else:
#         samples = sample_definitions[sample_name]
#     if mincov:
#         data = data.apply_coverage_limit(mincov)
#     if minsize:
#         data = data.apply_size_limit(minsize)
#     filepath = pattern.format(
#         mincov=mincov, minsize=minsize, show_values=show_values, level=level,
#         sample_name=sample_name)
#     print(f'Creating {filepath}')
#     corr_heatmap_with_size_partial(data, filepath=filepath, samples=samples,
#                                    add_values=show_values)
#
#
#
#
#
#
#
#
# corr_heatmap(rep_meth_filtered, filepath=corr_heatmap_fp_no_treat, samples=['WT', 'KI'], height_cm=10, width_cm=10,
#              add_values=)
# corr_heatmap(rep_meth, filepath=corr_heatmap_fp_nofilter, samples=None, height_cm=10, width_cm=10, add_values=)
# corr_heatmap(rep_meth, filepath=corr_heatmap_fp_no_treat_nofilter, samples=['WT', 'KI'], height_cm=10, width_cm=10,
#              add_values=)
#

