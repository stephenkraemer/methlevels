from itertools import chain

print('reloaded')
#-
from typing import Optional, List, Callable, Union

import matplotlib
matplotlib.use('Agg') # import before pyplot import!
import numpy as np
import pandas as pd
from pandas import IndexSlice as idxs

from methlevels.utils import NamedColumnsSlice as ncls
#-



class MethStats:
    """

    - chromosome name will be changed to chr, chrom and #chrom as input also allowed
    - uses pop order from df if none given, currently not save as attribute, only used in this function
    - names: beta, n_meth, n_total; also recognizes beta_value, renames to beta

    rep level amy currently be string or int

    chromosome turned into categorical if not categorical. if chormosome order given, using it

    from flat separates anno and data
    passing hierarchical: must separate self
    """

    grange_col_name_mapping = {
        'chrom': 'Chromosome',
        '#chrom': 'Chromosome',
        '#chr': 'Chromosome',
        'chr': 'Chromosome',
        'Chromosome': 'Chromosome',
        'start': 'Start',
        'end': 'End',
    }
    grange_col_names = ['Chromosome', 'Start', 'End']
    _all_column_names = ['Subject', 'Replicate', 'Stat']
    meth_stat_names = ['n_meth', 'n_total', 'beta_value']


    def __init__(self, meth_stats: pd.DataFrame, anno: Optional[pd.DataFrame] = None,
                 chromosome_order: Optional[List[str]] = None,
                 subject_order: Optional[List[str]] = None) -> None:

        assert isinstance(meth_stats.columns, pd.MultiIndex), \
            'This dataframe has a flat column index. Please use the dedicated constructor function'

        if anno is not None:
            assert meth_stats.index.equals(anno.index), 'meth_stats and anno must have same index'
        self._meth_stats = meth_stats.copy()
        self._anno = anno.copy() if anno is not None else None
        self.chromosome_order = list(chromosome_order) if chromosome_order else None
        self.subject_order = list(subject_order) if subject_order else None

        self.level = 'Replicate' if len(meth_stats.columns.levels) == 3 else 'Subject'
        self.column_names = self._all_column_names if self.level == 'Replicate' else [
            self._all_column_names[0], self._all_column_names[2]
        ]

        self._process_hierarchical_dataframe()

    @property
    def df(self) -> pd.DataFrame:
        return self._meth_stats


    @df.setter
    def df(self, data):
        self._meth_stats = data
        if self._anno is not None:
            assert not self._meth_stats.index.duplicated().any()
            self._anno = self._anno.loc[data.index]


    @property
    def anno(self):
        return self._anno


    @anno.setter
    def anno(self, anno):
        assert not anno.index.duplicated().any()
        self._anno = anno
        self._meth_stats = self._meth_stats.loc[anno.index, :]


    @classmethod
    def from_flat_dataframe(cls, meth_stats_with_anno: pd.DataFrame,
                            pop_order: List[str]=None,
                            additional_index_cols: List[str] = None,
                            drop_additional_index_cols: bool = True):
        """expects integer index and columns

        may contain anno cols

        matches both subject and replicate level

        input data
        - chromosome column must be categorical
        - various names for gr columns are allowed, will be converted to standard

        if order not given, order of columns at input will be used to make subject index categorical

        replicate is currently a string

        """

        meth_stats_with_anno = meth_stats_with_anno.copy()

        meth_stats_with_anno.rename(columns=cls.grange_col_name_mapping, inplace=True)
        assert meth_stats_with_anno['Start'].dtype == np.int64
        assert meth_stats_with_anno['End'].dtype == np.int64

        if additional_index_cols is not None:
            index_levels = cls.grange_col_names + additional_index_cols
            index_levels_to_drop = (index_levels if drop_additional_index_cols
                                    else cls.grange_col_names)
        else:
            index_levels = cls.grange_col_names
            index_levels_to_drop = cls.grange_col_names

        gr_index = meth_stats_with_anno.set_index(index_levels).index
        assert not gr_index.duplicated().any()
        meth_stats_with_anno = meth_stats_with_anno.drop(index_levels_to_drop, axis=1)

        rep_level_meth_stat_regex = (rf"([\w-]+)"
                                     rf"_(\d+|all)"
                                     f"_(beta_value|n_meth|n_total)")
        pop_level_meth_stat_regex = (rf"([\w-]+)"
                                     f"_(beta_value|n_meth|n_total)")
        column_names_ser = meth_stats_with_anno.columns.to_series()
        if column_names_ser.str.match(rep_level_meth_stat_regex).any():
            print('Recognized replicate level data')
            level = 'rep'
            metadata_df = column_names_ser.str.extract(rep_level_meth_stat_regex, expand=True)
        else:
            print('Recognized subject level data')
            level = 'pop'
            metadata_df = column_names_ser.str.extract(pop_level_meth_stat_regex, expand=True)

        is_anno_col = metadata_df.isna().all(axis=1)

        if is_anno_col.sum():
            meth_stats = meth_stats_with_anno.loc[:, ~is_anno_col]
            anno = meth_stats_with_anno.loc[:, is_anno_col]
        else:
            meth_stats = meth_stats_with_anno
            anno = None

        meth_stats_pops = metadata_df.iloc[~is_anno_col.values, 0]
        if pop_order is not None:
            if not set(meth_stats_pops) == set(pop_order):
                raise ValueError('Populations in meth stats and in pop order'
                                 'vector dont match.'
                                 f'Meth stats: {meth_stats_pops}'
                                 f'Pop order: {pop_order}')
        else:
            pop_order = list(meth_stats_pops.unique())

        pop_categorical = pd.Categorical(
                metadata_df.iloc[~is_anno_col.values, 0].values,
                categories=pop_order, ordered=True)

        multi_idx_cols = [pop_categorical, metadata_df.iloc[~is_anno_col.values, 1]]
        if level == 'rep':
            multi_idx_cols.append(metadata_df.iloc[~is_anno_col.values, 2])
            column_names = cls._all_column_names
        else:
            column_names = [cls._all_column_names[0], cls._all_column_names[2]]
        multi_idx = pd.MultiIndex.from_arrays(arrays=multi_idx_cols, names=column_names)

        meth_stats.columns = multi_idx

        meth_stats.index = gr_index
        meth_stats = meth_stats.sort_index(axis=0).sort_index(axis=1)
        if anno is not None:
            anno.index = gr_index
            anno = anno.sort_index(axis=0)

        return cls(meth_stats=meth_stats, anno=anno)


    def add_beta_values(self):
        """will add new or overwrite existing beta values

        NB: implementation differs depending on whether this is an update
        or new creation task. this is determiend by checking whether beta_value
        is a Statistic index level. This will lead to inconsistent results if some,
        but not all (Subject, Replicate) groups have beta_values

        """

        if self.level == 'Replicate':
            group_cols = [self._all_column_names[0], self._all_column_names[1]]
        else:
            group_cols = [self._all_column_names[0]]

        def add_beta_values(group_df):
            df = group_df.copy()
            idx = df.columns[0][0:-1]
            df.loc[:, idx + ('beta_value', )] = (
                    df.loc[:, idx + ('n_meth',) ] / df.loc[:, idx + ('n_total', )])
            return df

        if 'beta_value' in self.df.columns.levels[-1]:
            self.df = self.df.groupby(level=group_cols, axis=1, group_keys=False).apply(add_beta_values)
        else:
            # pandas 0.23 does not allow new columns in axis=1 free groupby-apply
            # noinspection PyAttributeOutsideInit
            self.df = pd.concat([add_beta_values(group_df)
                                 for name, group_df in self.df.groupby(level=group_cols, axis=1)], axis=1)


    def update(self, df: Optional[Union[pd.DataFrame, Callable]] = None,
               anno: Optional[Union[pd.DataFrame, Callable]] = None,
               inplace: bool = False) -> 'MethStats':
        """Can only assign to variable at a time, the other will be synchronized

        Returns new instance
        """
        # need to change indexing mechanism
        if df is not None and anno is not None:
            raise ValueError('Can only update one dataset at a time')
        elif df is not None:
            if callable(df):
                df = df(self.df)
            if inplace:
                # noinspection PyAttributeOutsideInit
                self.df = df
                return self
            else:
                return MethStats(meth_stats=df, anno=self.anno.loc[df.index, :])
        elif anno is not None:
            if callable(anno):
                anno = anno(self.anno)
            if inplace:
                # noinspection PyAttributeOutsideInit
                self.anno = anno
                return self
            else:
                return MethStats(meth_stats=self.df.loc[anno.index, :], anno=anno)
        else:
            raise TypeError('No update specified')


    def _process_hierarchical_dataframe(self):
        """Expects GR index and hierarchical column index

        if subject string, will be turned to categorical in order of appearance
        or in pop_order

        names must be subject, rep, ...
        must be non empty
        different grange names allowed

        i64 for start, end; categorical for chr, subject
        """

        df = self._meth_stats

        assert not df.isnull().all().all(), 'All meth stats are NA'

        df.index.rename(self.grange_col_names + df.index.names[3:], inplace=True)
        assert set(self.grange_col_names ) <= set(df.index.names)

        assert df.index.get_level_values('Start').dtype == np.int64
        assert df.index.get_level_values('End').dtype == np.int64

        chromosome_index = df.index.get_level_values('Chromosome')
        if chromosome_index.dtype.name != 'category':
            if not isinstance(chromosome_index[0], str):
                print("WARNING: passing integer chromosome index is currently inefficient")
                chromosome_index = chromosome_index.astype(str)
            if self.chromosome_order is not None:
                chromosome_order = self.chromosome_order
            else:
                # otherwise will carry over 'Chromosome' name into categorical data structure
                chromosome_order = chromosome_index.unique().values
            # otherwise will carry over 'Chromosome' name into categorical data structure
            chromosome_index = pd.Categorical(list(chromosome_index.values),
                                              categories=chromosome_order, ordered=True)
            print(chromosome_index.categories)
            other_index_levels = (df.index.get_level_values(i) for i in range(1, df.index.nlevels))
            row_midx = pd.MultiIndex.from_arrays(chain([chromosome_index], other_index_levels),
                                                 names=self.df.index.names)
            df.index = row_midx
            if self.anno is not None:
                self.anno.index = row_midx
        else:
            assert isinstance(chromosome_index.categories[0], str)
        if len(df.columns.levels) == 2:
            df.columns.names = self.column_names
        else:
            df.columns.names = self.column_names
        subject_idx = df.columns.get_level_values('Subject')
        if not subject_idx.dtype.name == 'category':
            subject_order = self.subject_order if self.subject_order else list(df.columns.get_level_values(0).unique())
            index_arrays = [
                pd.Categorical(subject_idx, categories=subject_order, ordered=True),
                df.columns.get_level_values(1)
            ]
            if len(df.columns.levels) == 3:
                index_arrays.append(df.columns.get_level_values(2))
                names = self.column_names
            else:
                names = self.column_names
            col_midx = pd.MultiIndex.from_arrays(
                    arrays=index_arrays,
                    names=names)
            df.columns = col_midx

        if not df.index.is_lexsorted():
            df.sort_index(inplace=True, axis=0)
        if not df.columns.is_lexsorted():
            df.sort_index(inplace=True, axis=1)

        self._meth_stats = df


    def apply_coverage_limit(self, threshold: int):
        has_enough_coverage = (self._meth_stats.loc[:, idxs[:, :, 'n_total']] >= threshold).all(axis=1)
        new_meth_stats = self._meth_stats.loc[has_enough_coverage, :].copy()
        if self._anno is None:
            new_anno = None
        else:
            new_anno = self._anno.loc[has_enough_coverage, :].copy()
        return MethStats(meth_stats=new_meth_stats, anno=new_anno)

    def apply_fraction_loss_capped_coverage_limit(
            self, coverage_threshold, max_fraction_lost):


        coverage_stats = self._meth_stats.loc[:, ncls(stat='n_total')]
        per_sample_quantiles = coverage_stats.quantile(max_fraction_lost, axis=0)
        per_sample_thresholds = (per_sample_quantiles
                                 .where(per_sample_quantiles < coverage_threshold,
                                        coverage_threshold))
        coverage_mask: pd.DataFrame = coverage_stats >= per_sample_thresholds
        coverage_ok_in_all_samples = coverage_mask.all(axis=1)
        new_meth_stats = self._meth_stats.loc[coverage_ok_in_all_samples, :].copy()
        if self._anno is None:
            new_anno = None
        else:
            new_anno = self._anno.loc[coverage_ok_in_all_samples, :].copy()
        return MethStats(meth_stats=new_meth_stats, anno=new_anno)


    def apply_size_limit(self, n_cpg_threshold):
        is_large_enough = self._anno.loc[:, 'n_cpg'] >= n_cpg_threshold
        new_meth_stats = self._meth_stats.loc[is_large_enough, :].copy()
        if self._anno is None:
            new_anno = None
        else:
            new_anno = self._anno.loc[is_large_enough, :].copy()
        return MethStats(meth_stats=new_meth_stats, anno=new_anno)


    def convert_to_populations(self):
        pop_stats_df = self._meth_stats.groupby(level=['Subject', 'Stat'], axis=1).sum()

        def update_beta_values(group_df):
            group_df[group_df.name, 'beta_value'] = (group_df[group_df.name, 'n_meth']
                                                     / group_df[group_df.name, 'n_total'])
            return group_df
        pop_stats_df = pop_stats_df.groupby(level='Subject', axis=1).apply(update_beta_values)

        if self._anno is None:
            anno_copy = None
        else:
            anno_copy = self._anno.copy()

        return MethStats(meth_stats=pop_stats_df, anno=anno_copy)


    def sample(self, n, random_state=123):
        if 0 < n < self._meth_stats.shape[0]:
            new_meth_stats = self._meth_stats.sample(n, random_state=random_state).sort_index()
            new_anno = self._anno.loc[new_meth_stats.index, :]
            return MethStats(meth_stats=new_meth_stats, anno=new_anno)
        else:
            raise ValueError('More sampled features requested than nrows')

    def get_most_var_betas(self, n):
        if 0 < n < self._meth_stats.shape[0]:
            most_var_idx = (self._meth_stats
                            .loc[:, ncls(stat='beta_value')]
                            .var(axis=1)
                            .sort_values(ascending=False)
                            .iloc[0:n]
                            .index
                            )
            new_meth_stats = self._meth_stats.loc[most_var_idx, :].sort_index()
            new_anno = self._anno.loc[new_meth_stats.index, :]
            return MethStats(meth_stats=new_meth_stats, anno=new_anno)
        return self

    def choose_n_observations(self, n, method='sample', random_state=123):
        if method=='sample':
            return self.sample(n, random_state)
        elif method=='mostvar':
            return self.get_most_var_betas(n)
        else:
            raise ValueError(f'Unknown sampling method {method}')

    def __str__(self):
        return str(self._meth_stats.head())

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

            betas = meth_in_regions._meth_stats.loc[:, ncls(stat='beta_value')]

            abs_deltas = betas.max(axis=1) - betas.min(axis=1)
            has_sufficient_delta = abs_deltas > min_delta

            meth_in_regions = MethStats(
                    meth_stats=meth_in_regions._meth_stats.loc[has_sufficient_delta, :].copy(),
                    anno=meth_in_regions._anno.loc[has_sufficient_delta, :].copy(),
            )

        return meth_in_regions

    def get_flat_columns(self):
        return ['_'.join(x) for x in self._meth_stats.columns]

    def save(self, filepaths, pop_name_abbreviations=None):
        multi_idx = self._meth_stats.columns
        if pop_name_abbreviations:
            reverse_abbreviation_mapping = {v:k for k,v in pop_name_abbreviations.items()}
            self._meth_stats = self._meth_stats.rename(reverse_abbreviation_mapping,
                                                       axis=1, level=0)
        self._meth_stats.columns = self.get_flat_columns()
        res = pd.concat([self._anno, self._meth_stats], axis=1).reset_index()
        for fp in filepaths:
            if fp.endswith('.feather'):
                res.to_feather(fp)
            elif fp.endswith('.tsv'):
                res.to_csv(fp, sep='\t', header=True, index=False)
            elif fp.endswith('.p'):
                res.to_pickle(fp)
            else:
                ValueError(f'Unknown suffix for {fp}, abort saving')
        self._meth_stats.columns = multi_idx



