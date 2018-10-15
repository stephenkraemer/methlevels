from itertools import chain

print('reloaded')
#-
from typing import Optional, List, Callable, Union, Dict, Tuple

import matplotlib
matplotlib.use('Agg') # import before pyplot import!
import numpy as np
import pandas as pd
from pandas import IndexSlice as idxs

from methlevels.utils import NamedColumnsSlice as ncls
from methlevels.utils import NamedIndexSlice as nidxs
from methlevels import gr_names
#-

# TODO: additional region identifier cols are not in the tidy format contracts
def _assert_tidy_meth_stats_data_contract(df):
    """

    observation = genomic interval[, additional genomic interval identifier], population[, replicate]
    variables = beta_value n_meth n_total ...
    first columns are genomic interval columns, the the population and replicate columns,
    then variables

    Uses simple RangeIndex. This is generally used for plotting functions,
    less for data manipulation which is better done on the standard wide-format.
    For these functions, it is often more convenient to not separate the
    index variables into a multiindex.
    """
    # First three columns are GRange columns, then Subject[, Replicate]

    assert df.columns.name is None
    assert isinstance(df.index, pd.RangeIndex)
    assert list(df.columns[0:3]) == gr_names.all
    assert df[gr_names.chrom].dtype.name == 'category'
    assert isinstance(df[gr_names.chrom].iloc[0], str)
    assert df[gr_names.start].dtype ==  df[gr_names.end].dtype == np.int64
    assert df.columns.contains('Subject')

    # Sorted by chrom, start, end
    assert df[gr_names.chrom].is_monotonic
    assert df[gr_names.start].is_monotonic
    assert df[gr_names.end].is_monotonic
    assert all(s in df.columns for s in MethStats.meth_stat_names)

    return True

def _assert_tidy_anno_contract(anno, df):
    """

    see _assert_tidy_meth_stats_data_contract for description of tidy
    format

    anno is always aligned with a meth stats df (ie describes the same
    genomic intervals in the same order). It does not have subject or
    replicate index variables, because the annotations only depend on the
    genomic interval. Therefore, meth_stats_df.shape[0] = anno.shape[0] * n_subjects * n_replicates
    """

    assert anno.columns.name is None
    columns = anno.columns

    # Index variables are GRange cols, same for all samples,
    # so no Subject, Replicate info
    assert list(columns[0:3]) == gr_names.all
    assert not columns.contains('Subject')

    # Sorted by chrom, start, end - this aligns it with the GRange sorting order
    # of the meth stats df
    assert not columns.contains('Replicate')
    chrom_start_idx = [gr_names.chrom, gr_names.start]
    a =  df[chrom_start_idx].drop_duplicates().reset_index(drop=True)
    b = anno[[gr_names.chrom, gr_names.start]]
    assert a.equals(b)

    # meth stats df has grange info for each sample
    if df.columns.contains('Replicate'):
        n_samples = df[['Subject', 'Replicate']].drop_duplicates().shape[0]
    else:
        n_samples = df['Subject'].nunique()
    assert df.shape[0] == anno.shape[0] * n_samples


    return True


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

    stats: Dict[str, pd.DataFrame]

    """Note on contract enforcement
    
    All methods either return a new instance (almost always the case)
    or update df / anno in place through the properties. By calling
    the contract assertions at the end of the constructor function and
    at the end of the property setters, the contract should be reliably
    enforced
    """

    @staticmethod
    def _assert_meth_stats_data_contract(df):
        index = df.index
        assert index.names[0:3] == gr_names.all
        # Index may not have duplicates
        # if GRanges contain duplicates, an additional index level must
        # be used, e.g. specifying a region ID
        assert not index.duplicated().any()
        # Chromosome must be categorical and string
        assert index.get_level_values(0).dtype.name == 'category'
        assert isinstance(index.get_level_values(0).categories[0], str)
        # Start and end are int
        assert index.get_level_values(1).dtype == index.get_level_values(2).dtype == np.int64
        assert index.is_lexsorted()

        columns = df.columns
        # May have 2 or 3 levels: subject or replicate level resolution
        assert (columns.names == MethStats._all_column_names
                or columns.names == [MethStats._all_column_names[i] for i in [0, 2]])
        # Subject is categorical (for sorting, plotting...)
        assert columns.get_level_values(0).dtype.name == 'category'
        assert columns.get_level_values('Stat').dtype.name == 'object'
        # Stat must contain n_meth, n_total, may contain other columns
        assert {'n_meth', 'n_total'} <= set(columns.levels[-1])
        assert df.dtypes.loc[nidxs(Stat='n_meth')].eq(np.int).all()
        assert df.dtypes.loc[nidxs(Stat='n_total')].eq(np.int).all()
        assert columns.is_lexsorted()

    @staticmethod
    def _assert_anno_contract(anno, df):
        if anno is not None:
            assert anno.index.equals(df.index)
            assert not isinstance(anno.columns, pd.MultiIndex)
        if anno is not None and 'region_id' in anno:
            # assert that region IDs are not reused for different chromosomes
            assert not (anno['region_id'].groupby('Chromosome', group_keys=False)
                        .apply(lambda df: df.drop_duplicates())
                        .duplicated()
                        .any()
                        )


    def __init__(self, meth_stats: Optional[pd.DataFrame] = None,
                 anno: Optional[pd.DataFrame] = None,
                 element_meth_stats: Optional[pd.DataFrame] = None,
                 element_anno: Optional[pd.DataFrame] = None,
                 chromosome_order: Optional[List[str]] = None,
                 subject_order: Optional[List[str]] = None) -> None:

        self._assert_init_argument_properties(element_anno, element_meth_stats, meth_stats)

        self.level = self._determine_level(element_meth_stats, meth_stats)

        if meth_stats is not None and element_meth_stats is not None:
            assert meth_stats.columns.equals(element_meth_stats.columns)

        self.chromosome_order = list(chromosome_order) if chromosome_order else None
        self.subject_order = list(subject_order) if subject_order else None

        self.column_names = self._all_column_names if self.level == 'Replicate' else [
            self._all_column_names[0], self._all_column_names[2]
        ]

        if meth_stats is not None:
            self._meth_stats = self._process_hierarchical_dataframe(meth_stats)
            self._assert_meth_stats_data_contract(self._meth_stats)
        else:
            self._meth_stats = None

        if element_meth_stats is not None:
            self.element_meth_stats = self._process_hierarchical_dataframe(element_meth_stats)
            self._assert_meth_stats_data_contract(self.element_meth_stats)
        else:
            self.element_meth_stats = None

        if anno is not None:
            anno = anno.copy(deep=True)
            assert anno.index.equals(meth_stats.index)
            anno.index = self._meth_stats.index
            self._assert_anno_contract(anno, self._meth_stats)
            self._anno = anno.copy()
        else:
            self._anno = None

        # must be called after adding the anno attribute, because the df property
        # asserts compatibility between anno and df
        if meth_stats is not None and not self._meth_stats.columns.levels[-1].contains('beta_value'):
            self.add_beta_values()

        if element_anno is not None:
            element_anno = element_anno.copy(deep=True)
            assert element_anno.index.equals(element_meth_stats.index)
            element_anno.index = self.element_meth_stats.index
            self._assert_anno_contract(element_anno, self.element_meth_stats)
            self.element_anno = element_anno.copy()
        else:
            self.element_anno = None


        self.stats = dict()


    @staticmethod
    def _determine_level(element_meth_stats, meth_stats) -> str:
        # Determine level
        level = None
        if meth_stats is not None:
            level = 'Replicate' if len(meth_stats.columns.levels) == 3 else 'Subject'
        if element_meth_stats is not None:
            element_meth_stats_level = 'Replicate' if len(element_meth_stats.columns.levels) == 3 else 'Subject'
            if level:
                assert level == element_meth_stats_level
            else:
                level = element_meth_stats_level
        if level is None:
            raise TypeError('Must pass at least one of meth_stats and element_meth_stats')
        return level

    @staticmethod
    def _assert_init_argument_properties(element_anno, element_meth_stats, meth_stats):
        # General assertions
        if meth_stats is None and element_meth_stats is None:
            raise TypeError('Must pass at least one of meth_stats or element_meth_stats')
        if meth_stats is not None:
            assert isinstance(meth_stats.columns, pd.MultiIndex), \
                'This dataframe has a flat column index. Please use the dedicated constructor function'
        if element_meth_stats is not None:
            assert isinstance(element_meth_stats.columns, pd.MultiIndex), \
                'This dataframe has a flat column index. Please use the dedicated constructor function'
            assert element_anno is not None
            assert 'region_id' in element_anno

    @property
    def counts(self) -> pd.DataFrame:
        return self._meth_stats

    @counts.setter
    def counts(self, data):
        print('Deprecation warning: will change to counts and stats in the future')
        self._meth_stats = data
        if self._anno is not None:
            assert not self._meth_stats.index.duplicated().any()
            self._anno = self._anno.loc[data.index]
        self._assert_meth_stats_data_contract(self._meth_stats)
        self._assert_anno_contract(self._anno, self._meth_stats)

    @property
    def df(self) -> pd.DataFrame:
        print('Deprecation warning: will change to counts and stats in the future')
        return self._meth_stats


    @df.setter
    def df(self, data):
        print('Deprecation warning: will change to counts and stats in the future')
        self._meth_stats = data
        if self._anno is not None:
            assert not self._meth_stats.index.duplicated().any()
            self._anno = self._anno.loc[data.index]
        self._assert_meth_stats_data_contract(self._meth_stats)
        self._assert_anno_contract(self._anno, self._meth_stats)


    @property
    def anno(self):
        return self._anno


    @anno.setter
    def anno(self, anno):
        assert not anno.index.duplicated().any()
        self._anno = anno
        self._meth_stats = self._meth_stats.loc[anno.index, :]
        self._assert_meth_stats_data_contract(self._meth_stats)
        self._assert_anno_contract(self._anno, self._meth_stats)


    @classmethod
    def from_flat_dataframe(cls, meth_stats_with_anno: pd.DataFrame,
                            pop_order: List[str]=None,
                            elements: bool = False,
                            additional_index_cols: List[str] = None,
                            drop_additional_index_cols: bool = True):
        """expects integer index and columns
        
        Args:
            additional_index_cols: columns to be added to the index 
                (append to GRange index cols). Must be provided if the
                GRange index is non-unique.
            drop_additional_index_cols: wether to remove index columns
            from the value columns when they are set as index level

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

        # TODO: add test
        if elements:
            return cls(element_meth_stats=meth_stats, element_anno=anno)
        else:
            return cls(meth_stats=meth_stats, anno=anno)


    def to_tidy_format(self):
        if self.anno is not None:
            tidy_anno = (self.anno
                         # The wide format anno df may have the same data as index and value column
                         # -> remove
                         .drop(list(set(self.anno.index.names) & set(self.anno.columns)), axis=1)
                         .reset_index())
        else:
            tidy_anno = None
        tidy_df = self.df.stack(self.df.columns.names[0:-1]).reset_index()
        tidy_df.columns.name = None
        _assert_tidy_meth_stats_data_contract(tidy_df)
        if self.anno is not None:
            _assert_tidy_anno_contract(tidy_anno, tidy_df)
        return tidy_df, tidy_anno


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
            new_df = pd.concat([add_beta_values(group_df)
                                for name, group_df in self.df.groupby(level=group_cols, axis=1)], axis=1)
            new_df.sort_index(axis=1, inplace=True)
            # noinspection PyAttributeOutsideInit
            self.df = new_df


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


    def _process_hierarchical_dataframe(self, df):
        """Expects GR index and hierarchical column index

        if subject string, will be turned to categorical in order of appearance
        or in pop_order

        names must be subject, rep, ...
        must be non empty
        different grange names allowed

        i64 for start, end; categorical for chr, subject
        """

        df = df.copy(deep=True)

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
                                                 names=df.index.names)
            df.index = row_midx
        else:
            # noinspection PyUnresolvedReferences
            assert isinstance(chromosome_index.categories[0], str)
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

        return df

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


        coverage_stats = self._meth_stats.loc[:, ncls(Stat='n_total')]
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
        if self._meth_stats is not None:
            pop_stats_df = self._meth_stats.groupby(level=['Subject', 'Stat'], axis=1).sum()

            def update_beta_values(group_df):
                group_df[group_df.name, 'beta_value'] = (group_df[group_df.name, 'n_meth']
                                                         / group_df[group_df.name, 'n_total'])
                return group_df
            pop_stats_df = pop_stats_df.groupby(level='Subject', axis=1).apply(update_beta_values)

            pop_stats_df.loc[:, ncls(Stat='n_meth')] = pop_stats_df.loc[:, ncls(Stat='n_meth')].astype(np.int64)
            pop_stats_df.loc[:, ncls(Stat='n_total')] = pop_stats_df.loc[:, ncls(Stat='n_total')].astype(np.int64)

            if self._anno is None:
                anno_copy = None
            else:
                anno_copy = self._anno.copy()
        else:
            pop_stats_df = None
            anno_copy = None

        if self.element_meth_stats is not None:
            element_pop_stats_df = self.element_meth_stats.groupby(level=['Subject', 'Stat'], axis=1).sum()

            def update_beta_values(group_df):
                group_df[group_df.name, 'beta_value'] = (group_df[group_df.name, 'n_meth']
                                                         / group_df[group_df.name, 'n_total'])
                return group_df
            element_pop_stats_df = element_pop_stats_df.groupby(level='Subject', axis=1).apply(update_beta_values)

            element_pop_stats_df.loc[:, ncls(Stat='n_meth')] = element_pop_stats_df.loc[:, ncls(Stat='n_meth')].astype(np.int64)
            element_pop_stats_df.loc[:, ncls(Stat='n_total')] = element_pop_stats_df.loc[:, ncls(Stat='n_total')].astype(np.int64)

            if self.element_anno is None:
                element_anno_copy = None
            else:
                element_anno_copy = self.element_anno.copy()
        else:
            element_pop_stats_df = None
            element_anno_copy = None

        return MethStats(meth_stats=pop_stats_df, anno=anno_copy,
                         element_meth_stats=element_pop_stats_df,
                         element_anno=element_anno_copy)


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

            betas = meth_in_regions._meth_stats.loc[:, ncls(Stat='beta_value')]

            abs_deltas = betas.max(axis=1) - betas.min(axis=1)
            has_sufficient_delta = abs_deltas > min_delta

            meth_in_regions = MethStats(
                    meth_stats=meth_in_regions._meth_stats.loc[has_sufficient_delta, :].copy(),
                    anno=meth_in_regions._anno.loc[has_sufficient_delta, :].copy(),
            )

        return meth_in_regions

    def get_flat_columns(self):
        return ['_'.join(x) for x in self._meth_stats.columns]

    def save_flat(self, filepaths: List[str]):
        """save as flat dataframe, prepend the anno columns"""

        assert isinstance(filepaths, list)

        # will be restored at the end
        orig_multi_idx = self._meth_stats.columns

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

        self._meth_stats.columns = orig_multi_idx

    # Stat computation methods
    # ==========================================================================

    def add_beta_value_stat(self) -> 'MethStats':
        """Add beta-value stat df"""
        if self.level == 'Replicate':
            group_cols = [self._all_column_names[0], self._all_column_names[1]]
        else:
            group_cols = [self._all_column_names[0]]

        def add_beta_values(group_df):
            df = group_df.copy()
            idx = df.columns[0][0:-1]
            beta_values_ser = df.loc[:, idx + ('n_meth',) ] / df.loc[:, idx + ('n_total', )]
            return beta_values_ser

        beta_values = self.counts.groupby(level=group_cols, axis=1, group_keys=False, observed=True).apply(add_beta_values)
        self.stats['beta-value'] = beta_values

        return self


    def add_zscores(self, stat_name: str) -> 'MethStats':
        """Row-wise z-score normalization

        Args:
            stat_name: statistic as input for row-wise z-score
                normalization. Must already exist.

        Returns:
            self
        """
        df = self.stats[stat_name]
        self.stats[f'{stat_name}_zscores'] = (
            df.subtract(df.mean(axis=1), axis=0)
              .divide(df.std(axis=1), axis=0))
        return self

    def add_deltas(self, stat_name: str, root: Union[str, Tuple],
                   output_name: Optional[str] = None) -> 'MethStats':
        """Add delta score

        Args:
            stat_name: input statistic. Must already exist.
            root: sample to use as root population. Value must be usable
                with .loc indexing to retrieve the sample from the input
                df
            output_name: Optional. Otherwise, the name will be:
                {stat_name}_delta_{root_name}. If root is a tuple, the fields
                will be converted to str and concatenated with underscores.
        """

        df = self.stats[stat_name]
        if output_name is None:
            if isinstance(root, tuple):
                root_name = '_'.join((str(x) for x in root))
            else:
                assert isinstance(root, str)
                root_name = root
            output_name = f'{stat_name}_delta_{root_name}'
        self.stats[output_name] = df.subtract(df.loc[:, root], axis=0)
        return self

    # element_meth_stats queries
    # ==========================================================================
    def aggregate_element_counts(self) -> 'MethStats':
        """Aggregate element counts into interval counts

        - aggregates n_meth and n_total based on the region_ids in the
          element_anno dataframe
        - computes beta values for the interval stats

        Returns:
            - index of the resulting counts df will have levels
              ['Chromosome', 'Start', 'End', 'region_id']
        """
        if self.anno is not None:
            new_gr_index = self.anno.index
        else:
            def get_region_interval(df):
                return pd.Series({'Chromosome': df['Chromosome'].iloc[0],
                                  'Start': df['Start'].iloc[0],
                                  'End': df['End'].iloc[-1],
                                  'region_id': df['region_id'].iloc[0],
                                  'n_elements': len(df),
                                  'n_cpg': len(df)})
            # Possible problem: what happens if region_id is also in the index?
            new_anno_df = (self
                         .element_anno
                         .reset_index()
                         .groupby('region_id')
                         .apply(get_region_interval)
                         )
            new_anno_df['Chromosome'] = new_anno_df['Chromosome'].astype(
                    self.element_anno.index.get_level_values('Chromosome').dtype)
            new_anno_df.set_index(['Chromosome', 'Start', 'End', 'region_id'], inplace=True)
            new_gr_index = new_anno_df.index

        meth_stats_df = (self.element_meth_stats
                         .groupby(self.element_anno['region_id'])
                         .sum()
                         .set_axis(new_gr_index, axis=0, inplace=False)
                         )
        # noinspection PyAttributeOutsideInit
        self.counts = meth_stats_df

        # Add anno df after counts df, because the anno property contract
        # required infos from the counts df
        if self.anno is None:
            # noinspection PyAttributeOutsideInit,PyUnboundLocalVariable
            self.anno = new_anno_df

        if 'beta_value' in meth_stats_df.columns.get_level_values(-1):
            self.add_beta_values()

        return self


    def __eq__(self, other) -> bool:
        def compare_none_or_df(elem, other):
            if elem is None:
                return elem == other
            elif isinstance(elem, pd.DataFrame):
                return elem.equals(other)
            else:
                raise ValueError()
        return (compare_none_or_df(self.counts, other.counts)
                and compare_none_or_df(self.anno, other.anno)
                and compare_none_or_df(self.element_meth_stats,
                                       other.element_meth_stats)
                and compare_none_or_df(self.element_anno,
                                       other.element_anno)
                and self.level == other.level)



