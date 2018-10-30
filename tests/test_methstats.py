from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.api.types import CategoricalDtype
from pandas.util.testing import assert_index_equal

idxs = pd.IndexSlice

import pytest

import methlevels as ml
from methlevels import MethStats
from methlevels.utils import read_csv_with_padding
from methlevels.utils import read_csv_with_padding
from methlevels.utils import NamedIndexSlice as nidxs


# beta values here are amazingly wrong, but this is used for tests
# where it doesn't matter. exchange with new_flat_meth_stats later
@pytest.fixture()
def flat_meth_stats():
    return pd.DataFrame({
        'chr': pd.Categorical('1 1 2'.split(), categories=['1', '2'], ordered=True),
        'start': [1, 3, 5],
        'end': [2, 4, 6],
        'hsc_1_n_meth': [10, 10, 10],
        'hsc_1_n_total': [10, 10, 10],
        'hsc_1_beta_value': [.5, .5, .5],
        'hsc_2_n_meth': [10, 15, 10],
        'hsc_2_n_total': [10, 10, 10],
        'hsc_2_beta_value': [.5, .5, .5],
        'mpp1_1_n_meth': [10, 10, 10],
        'mpp1_1_n_total': [10, 10, 10],
        'mpp1_1_beta_value': [.5, .5, .5],
    })

@pytest.fixture()
def new_flat_meth_stats():
    return pd.DataFrame({
        'chr': pd.Categorical('1 1 2'.split(), categories=['1', '2'], ordered=True),
        'start': [1, 3, 5],
        'end': [2, 4, 6],
        'hsc_1_n_meth': [5, 6, 7],
        'hsc_1_n_total': [10, 10, 10],
        'hsc_1_beta_value': [.5, .6, .7],
        'hsc_2_n_meth': [5, 10, 5],
        'hsc_2_n_total': [10, 10, 10],
        'hsc_2_beta_value': [.5, 1., .5],
        'mpp1_1_n_meth': [10, 10, 10],
        'mpp1_1_n_total': [10, 10, 10],
        'mpp1_1_beta_value': [1., 1., 1.],
    })

@pytest.fixture()
def new_flat_meth_stats_pop_level():
    return pd.DataFrame({
        'chr': pd.Categorical('1 1 2'.split(), categories=['1', '2'], ordered=True),
        'start': [1, 3, 5],
        'end': [2, 4, 6],
        'hsc_n_meth': [10, 16, 12],
        'hsc_n_total': [20, 20, 20],
        'hsc_beta_value': [.5, 16/20, 12/20],
        'mpp1_n_meth': [10, 10, 10],
        'mpp1_n_total': [10, 10, 10],
        'mpp1_beta_value': [1., 1., 1.],
    })



@pytest.fixture()
def flat_meth_stats_subject_level():
    return pd.DataFrame({
        'chr': pd.Categorical('1 1 2'.split(), categories=['1', '2'], ordered=True),
        'start': [1, 3, 5],
        'end': [2, 4, 6],
        'hsc_n_meth': [10, 10, 10],
        'hsc_n_total': [10, 10, 10],
        'hsc_beta_value': [.5, .5, .5],
        'mpp1_n_meth': [10, 10, 10],
        'mpp1_n_total': [10, 10, 10],
        'mpp1_beta_value': [.5, .5, .5],
    })


@pytest.fixture
def flat_meth_stats3():
    flat_meth_stats = read_csv_with_padding("""\
chr , start , end , region_id , anno1 , hsc_1_beta_value , hsc_1_n_meth , hsc_1_n_total , hsc_2_beta_value , hsc_2_n_meth , hsc_2_n_total , mpp1_1_beta_value , mpp1_1_n_meth , mpp1_1_n_total
1   , 1     , 3   , 0         , a    , 0.5              , 10           , 10            , 0.5              , 10           , 10            , 0.5             , 10          , 10
1   , 5     , 7   , 0         , a    , 0.5              , 10           , 10            , 0.5              , 10           , 10            , 0.5             , 10          , 10
1   , 13    , 15  , 1         , b    , 0.5              , 10           , 10            , 0.5              , 15           , 10            , 0.5             , 10          , 10
1   , 20    , 24  , 1         , b    , 0.5              , 10           , 10            , 0.5              , 15           , 10            , 0.5             , 10          , 10
2   , 50    , 60  , 2         , c    , 0.5              , 10           , 10            , 0.5              , 10           , 10            , 0.5             , 10          , 10
2   , 80    , 90  , 2         , c    , 0.5              , 10           , 10            , 0.5              , 10           , 10            , 0.5             , 10          , 10
    """)
    return flat_meth_stats


@pytest.fixture
def hierarchical_counts3():
    hierarchical_counts = read_csv_with_padding("""\
        Subject    ,       ,     ,           , hsc        , hsc    , hsc     , hsc        , hsc    , hsc     , mpp1       , mpp1   , mpp1
        Replicate  ,       ,     ,           , 1          , 1      , 1       , 2          , 2      , 2       , 1          , 1      , 1
        Stat       ,       ,     ,           , beta_value , n_meth , n_total , beta_value , n_meth , n_total , beta_value , n_meth , n_total
        Chromosome , Start , End , region_id ,            ,        ,         ,            ,        ,         ,            ,        ,
        1          , 1     , 3   , 0         , 0.5        , 10     , 10      , 0.5        , 10     , 10      , 0.5        , 10     , 10
        1          , 5     , 7   , 0         , 0.5        , 10     , 10      , 0.5        , 10     , 10      , 0.5        , 10     , 10
        1          , 13    , 15  , 1         , 0.5        , 10     , 10      , 0.5        , 15     , 10      , 0.5        , 10     , 10
        1          , 20    , 24  , 1         , 0.5        , 10     , 10      , 0.5        , 15     , 10      , 0.5        , 10     , 10
        2          , 50    , 60  , 2         , 0.5        , 10     , 10      , 0.5        , 10     , 10      , 0.5        , 10     , 10
        2          , 80    , 90  , 2         , 0.5        , 10     , 10      , 0.5        , 10     , 10      , 0.5        , 10     , 10
    """, index_col=[0, 1, 2, 3], header=[0, 1, 2])
    return hierarchical_counts


@pytest.fixture
def anno3():
    anno = read_csv_with_padding("""\
        Chromosome , Start , End , region_id , anno1
        1          , 1     , 3   , 0         , a
        1          , 5     , 7   , 0         , a
        1          , 13    , 15  , 1         , b
        1          , 20    , 24  , 1         , b
        2          , 50    , 60  , 2         , c
        2          , 80    , 90  , 2         , c
    """, index_col=[0, 1, 2, 3], header=[0])
    return anno


def test_anno_contract(flat_meth_stats):
    meth_stats = MethStats.from_flat_dataframe(flat_meth_stats)
    with pytest.raises(AssertionError):
        meth_stats.anno = pd.DataFrame([1, 2])

    with pytest.raises(AssertionError):
        meth_stats = MethStats.from_flat_dataframe(flat_meth_stats)
        MethStats(meth_stats=meth_stats.df, anno=pd.DataFrame([1, 2]))

def test_df_contract(flat_meth_stats):
    flat_meth_stats_duplicates = pd.concat([flat_meth_stats, flat_meth_stats])
    with pytest.raises(AssertionError):
        meth_stats = MethStats.from_flat_dataframe(flat_meth_stats_duplicates)
    with pytest.raises(AssertionError):
        MethStats(pd.concat([MethStats.from_flat_dataframe(flat_meth_stats).df,
                             MethStats.from_flat_dataframe(flat_meth_stats).df]))


@pytest.mark.parametrize('with_anno_cols', [True, False])
def test_initializes_from_flat_df(with_anno_cols, flat_meth_stats):

    if with_anno_cols:
        flat_meth_stats = flat_meth_stats.assign(size= [10, 12, 14],
                                                 n_cpg= [3, 4, 6])

    meth_stats = MethStats.from_flat_dataframe(flat_meth_stats)

    assert meth_stats.df.loc[('1', 3, 4), ('hsc', '2', 'n_meth')] == 15
    assert meth_stats.df.index.names == meth_stats.grange_col_names

    if with_anno_cols:
        assert meth_stats.anno['size'].eq([10, 12, 14]).all()


def test_initializes_from_flat_df_subject_level(flat_meth_stats_subject_level):
    meth_stats = MethStats.from_flat_dataframe(flat_meth_stats_subject_level)
    assert meth_stats.df.loc[('1', 3, 4), ('hsc', 'n_meth')] == 10



@pytest.mark.parametrize('chromosome_order', [None, ['1', '2']])
@pytest.mark.parametrize('subject_index_type', ['category', 'object'])
@pytest.mark.parametrize('level', ['subject', 'replicate'])
def test_initializes_from_hierarchical_df(chromosome_order, subject_index_type, level):

    hierarchical_meth_stats = pd.DataFrame(
            [
                [10, 10, .5, 10, 10, .5],
                [99, 10, .5, 10, 10, .5],
                [10, 10, .5, 10, 10, .5],
            ]
    )

    if subject_index_type == 'category':
        subject_idx = pd.Categorical(['hsc', 'hsc', 'hsc', 'b-cells', 'b-cells', 'b-cells'], ordered=True)
        col_names = MethStats._all_column_names
    else:
        subject_idx = ['hsc', 'hsc', 'hsc', 'b-cells', 'b-cells', 'b-cells']
        col_names = None

    col_midx = pd.MultiIndex.from_arrays([
        subject_idx,
        [1, 1, 1, 1, 1, 1],
        ['n_meth', 'n_total', 'beta_value'] * 2
    ],
            names=col_names
    )

    if chromosome_order is None:
        chromosome_index = pd.Categorical('1 1 2'.split(), categories=chromosome_order, ordered=True)
    else:
        chromosome_index = '1 1 2'.split()

    row_midx = pd.MultiIndex.from_arrays([chromosome_index, [1, 2, 3], [2, 3, 4]],
                                         names=MethStats.grange_col_names)

    hierarchical_meth_stats.index = row_midx
    hierarchical_meth_stats.columns = col_midx

    if level == 'subject':
        hierarchical_meth_stats.columns = hierarchical_meth_stats.columns.droplevel(1)

    meth_stats = MethStats(meth_stats=hierarchical_meth_stats, chromosome_order=chromosome_order)

    if level == 'subject':
        assert meth_stats.df.loc[('1', 2, 3), ('hsc', 'n_meth')] == 99
    else:
        assert meth_stats.df.loc[('1', 2, 3), ('hsc', 1, 'n_meth')] == 99
    if chromosome_order is not None:
        assert list(meth_stats.df.index.get_level_values('Chromosome').categories) == chromosome_order

    if subject_index_type == 'object':
        columns = meth_stats.df.columns
        assert columns.get_level_values('Subject').dtype.name == 'category'
        assert list(columns.get_level_values('Subject'))== subject_idx
        assert columns.is_lexsorted()


@pytest.mark.parametrize('with_anno_cols', [True, False])
@pytest.mark.parametrize('elements', [True, False])
def test_convert_to_populations(new_flat_meth_stats, new_flat_meth_stats_pop_level,
                                with_anno_cols, elements):
    if with_anno_cols:
        new_flat_meth_stats['size'] = np.arange(
                len(new_flat_meth_stats))
        new_flat_meth_stats_pop_level['size'] = np.arange(
                len(new_flat_meth_stats_pop_level))
    # if the data are element-level, the anno df must contain a region_id column
    if elements:
        new_flat_meth_stats['region_id'] = np.arange(
                len(new_flat_meth_stats))
        new_flat_meth_stats_pop_level['region_id'] = np.arange(
                len(new_flat_meth_stats_pop_level))

    # The expected result has either the counts or the element_meth_stats filled,
    # but not both
    if elements:
        expected_pop_level_meth_stats = ml.MethStats.from_flat_dataframe(
                new_flat_meth_stats_pop_level, elements=True)
        assert expected_pop_level_meth_stats.element_meth_stats is not None
    else:
        expected_pop_level_meth_stats = ml.MethStats.from_flat_dataframe(
                new_flat_meth_stats_pop_level, elements=False)
        assert expected_pop_level_meth_stats.element_meth_stats is None

    # Read flat meth stats either as element or interval, then convert
    computed_pop_level_meth_stats = (
        ml.MethStats.from_flat_dataframe(
                new_flat_meth_stats, elements=elements)
            .convert_to_populations())

    # Two dfs should be None, two should be filled
    assert expected_pop_level_meth_stats == computed_pop_level_meth_stats


def test_add_beta_values(flat_meth_stats, flat_meth_stats_subject_level):
    meth_stats = MethStats.from_flat_dataframe(flat_meth_stats)

    # add 'de novo'
    meth_stats.df = meth_stats.df.drop('beta_value', axis=1, level=2)
    meth_stats.df.columns = meth_stats.df.columns.remove_unused_levels()
    meth_stats.add_beta_values()
    assert meth_stats.df.loc[('1', 3, 4), ('hsc', '2', 'beta_value')] == 1.5

    # subject level
    meth_stats = MethStats.from_flat_dataframe(flat_meth_stats_subject_level)
    meth_stats.add_beta_values()
    assert meth_stats.df.loc[('1', 3, 4), ('hsc', 'beta_value')] == 1.0

    # update
    meth_stats = MethStats.from_flat_dataframe(flat_meth_stats)
    meth_stats.df.loc[('1', 3, 4), ('hsc', '2', 'n_meth')] = 3
    meth_stats.df.loc[('1', 3, 4), ('hsc', '2', 'n_total')] = 10
    meth_stats.add_beta_values()
    assert meth_stats.df.loc[('1', 3, 4), ('hsc', '2', 'beta_value')] == 0.3


def test_update(flat_meth_stats):

    flat_meth_stats['anno1'] = range(flat_meth_stats.shape[0])
    meth_stats = MethStats.from_flat_dataframe(flat_meth_stats)
    old_anno = meth_stats.anno.copy()
    old_df = meth_stats.df.copy()

    def assert_correct_update(old_meth_stats, new_meth_stats):
        assert_frame_equal(old_meth_stats.anno, old_anno)
        assert_frame_equal(old_meth_stats.df, old_df)
        assert_frame_equal(new_meth_stats.anno, old_anno.iloc[0:2])
        assert_frame_equal(new_meth_stats.df, old_df.iloc[0:2])

    # update df
    ## from values
    new_meth_stats = meth_stats.update(df = meth_stats.df.iloc[0:2])
    assert_correct_update(meth_stats, new_meth_stats)

    # with callable
    new_meth_stats = meth_stats.update(df = lambda df: df.iloc[0:2])
    assert_correct_update(meth_stats, new_meth_stats)

    # update anno
    ## with values
    new_meth_stats = meth_stats.update(anno = meth_stats.anno.iloc[0:2])
    assert_correct_update(meth_stats, new_meth_stats)

    ## with callable
    new_meth_stats = meth_stats.update(anno = lambda anno: anno.iloc[0:2])
    assert_correct_update(meth_stats, new_meth_stats)

    # update inplace
    ## df with callable
    new_meth_stats = deepcopy(meth_stats)
    new_meth_stats.update(df = lambda df: df.iloc[0:2], inplace=True)
    assert_correct_update(meth_stats, new_meth_stats)

    ## anno with value
    new_meth_stats = deepcopy(meth_stats)
    new_meth_stats.update(anno = meth_stats.anno.iloc[0:2], inplace=True)
    assert_correct_update(meth_stats, new_meth_stats)


def test_additional_index_column():

    flat_meth_stats = read_csv_with_padding("""\
chr , start , end , region_id , anno1 , hsc_1_beta_value , hsc_1_n_meth , hsc_1_n_total , hsc_2_beta_value , hsc_2_n_meth , hsc_2_n_total , mpp1_1_beta_value , mpp1_1_n_meth , mpp1_1_n_total
1   , 1     , 3   , 0         , a    , 0.5              , 10           , 10            , 0.5              , 10           , 10            , 0.5             , 10          , 10
1   , 5     , 7   , 0         , a    , 0.5              , 10           , 10            , 0.5              , 10           , 10            , 0.5             , 10          , 10
1   , 13    , 15  , 1         , b    , 0.5              , 10           , 10            , 0.5              , 15           , 10            , 0.5             , 10          , 10
1   , 20    , 24  , 1         , b    , 0.5              , 10           , 10            , 0.5              , 15           , 10            , 0.5             , 10          , 10
2   , 50    , 60  , 2         , c    , 0.5              , 10           , 10            , 0.5              , 10           , 10            , 0.5             , 10          , 10
2   , 80    , 90  , 2         , c    , 0.5              , 10           , 10            , 0.5              , 10           , 10            , 0.5             , 10          , 10
    """)

    meth_stats = ml.MethStats.from_flat_dataframe(flat_meth_stats,
                                                  additional_index_cols=['region_id'])

    expected_df = read_csv_with_padding("""\
        Subject    ,       ,     ,           , hsc        , hsc    , hsc     , hsc        , hsc    , hsc     , mpp1       , mpp1   , mpp1
        Replicate  ,       ,     ,           , 1          , 1      , 1       , 2          , 2      , 2       , 1          , 1      , 1
        Stat       ,       ,     ,           , beta_value , n_meth , n_total , beta_value , n_meth , n_total , beta_value , n_meth , n_total
        Chromosome , Start , End , region_id ,            ,        ,         ,            ,        ,         ,            ,        ,
        1          , 1     , 3   , 0         , 0.5        , 10     , 10      , 0.5        , 10     , 10      , 0.5        , 10     , 10
        1          , 5     , 7   , 0         , 0.5        , 10     , 10      , 0.5        , 10     , 10      , 0.5        , 10     , 10
        1          , 13    , 15  , 1         , 0.5        , 10     , 10      , 0.5        , 15     , 10      , 0.5        , 10     , 10
        1          , 20    , 24  , 1         , 0.5        , 10     , 10      , 0.5        , 15     , 10      , 0.5        , 10     , 10
        2          , 50    , 60  , 2         , 0.5        , 10     , 10      , 0.5        , 10     , 10      , 0.5        , 10     , 10
        2          , 80    , 90  , 2         , 0.5        , 10     , 10      , 0.5        , 10     , 10      , 0.5        , 10     , 10
    """, index_col=[0, 1, 2, 3], header=[0, 1, 2])

    expected_anno = read_csv_with_padding("""\
        Chromosome , Start , End , region_id , anno1
        1          , 1     , 3   , 0         , a
        1          , 5     , 7   , 0         , a
        1          , 13    , 15  , 1         , b
        1          , 20    , 24  , 1         , b
        2          , 50    , 60  , 2         , c
        2          , 80    , 90  , 2         , c
    """, index_col=[0, 1, 2, 3], header=[0])

    meth_stats_expected = ml.MethStats(meth_stats=expected_df, anno=expected_anno)

    assert_frame_equal(meth_stats.df, meth_stats_expected.df, check_names=False)
    assert_frame_equal(meth_stats.anno, meth_stats_expected.anno, check_names=False)

    meth_stats = ml.MethStats.from_flat_dataframe(flat_meth_stats,
                                                  additional_index_cols=['region_id'],
                                                  drop_additional_index_cols=False)

    expected_anno_no_drop = read_csv_with_padding("""\
        Chromosome , Start , End , region_id , region_id , anno1
        1          , 1     , 3   , 0         , 0         , a
        1          , 5     , 7   , 0         , 0         , a
        1          , 13    , 15  , 1         , 1         , b
        1          , 20    , 24  , 1         , 1         , b
        2          , 50    , 60  , 2         , 2         , c
        2          , 80    , 90  , 2         , 2         , c
    """, index_col=[0, 1, 2, 3], header=[0]).rename(columns={'region_id.1': 'region_id'})
    print(expected_anno_no_drop)

    meth_stats_expected_no_drop = ml.MethStats(meth_stats=expected_df, anno=expected_anno_no_drop)

    assert_frame_equal(meth_stats.df, meth_stats_expected_no_drop.df, check_names=False)
    assert_frame_equal(meth_stats.anno, meth_stats_expected_no_drop.anno, check_names=False)


@pytest.mark.parametrize('level', ['Subject', 'Replicate'])
@pytest.mark.parametrize('with_anno', [True, False])
def test_to_tidy_format(flat_meth_stats, level, with_anno):
    if with_anno:
        flat_meth_stats['anno'] = list('abc')
    meth_stats = MethStats.from_flat_dataframe(flat_meth_stats)
    if level == 'Subject':
        meth_stats = meth_stats.convert_to_populations()
    tidy_df, tidy_anno = meth_stats.to_tidy_format()

    dtypes_d = {'Chromosome': CategoricalDtype(['1', '2'], ordered=True),
                'Subject': CategoricalDtype(['hsc', 'mpp1'], ordered=True),
                'Replicate': str}

    expected_tidy_anno = read_csv_with_padding("""\
        Chromosome , Start , End , anno
        1          , 1     , 2   , a
        1          , 3     , 4   , b
        2          , 5     , 6   , c
        """, dtype=dtypes_d)

    if level == 'Subject':
        expected_tidy_df = read_csv_with_padding("""\
            Chromosome,Start,End,Subject,beta_value,n_meth,n_total
            1,1,2,hsc,1,20,20
            1,1,2,mpp1,1,10,10
            1,3,4,hsc,1.25,25,20
            1,3,4,mpp1,1,10,10
            2,5,6,hsc,1,20,20
            2,5,6,mpp1,1,10,10
            """, dtype=dtypes_d)
    else:
        expected_tidy_df = read_csv_with_padding("""\
            Chromosome , Start , End , Subject , Replicate , beta_value , n_meth , n_total
            1          , 1     , 2   , hsc     , 1         , 0.5        , 10.0   , 10.0
            1          , 1     , 2   , hsc     , 2         , 0.5        , 10.0   , 10.0
            1          , 1     , 2   , mpp1    , 1         , 0.5        , 10.0   , 10.0
            1          , 3     , 4   , hsc     , 1         , 0.5        , 10.0   , 10.0
            1          , 3     , 4   , hsc     , 2         , 0.5        , 15.0   , 10.0
            1          , 3     , 4   , mpp1    , 1         , 0.5        , 10.0   , 10.0
            2          , 5     , 6   , hsc     , 1         , 0.5        , 10.0   , 10.0
            2          , 5     , 6   , hsc     , 2         , 0.5        , 10.0   , 10.0
            2          , 5     , 6   , mpp1    , 1         , 0.5        , 10.0   , 10.0
            """, dtype=dtypes_d)

    assert_frame_equal(tidy_df, expected_tidy_df)
    if with_anno:
        assert_frame_equal(tidy_anno, expected_tidy_anno)
    else:
        assert tidy_anno is None


# Test stat creation
# ######################################################################

@pytest.mark.parametrize('remove_beta', [True, False])
def test_add_beta_value_stat(new_flat_meth_stats, remove_beta):
    meth_stats = MethStats.from_flat_dataframe(new_flat_meth_stats)
    if remove_beta:
        meth_stats.counts = meth_stats.counts.drop(columns=['beta_value'], level=-1)
    meth_stats.add_beta_value_stat()
    # need to retain one Stat to get the correct expected MIdx
    # beta_value is missing if remove_beta
    # n_total will always be there, so we drop beta_value and n_meth
    exp_columns_idx = (meth_stats.counts.columns
                       .drop(['n_meth', 'beta_value'], level=-1)
                       .droplevel(-1))
    expected_beta_values_df = pd.DataFrame([[.5, .5, 1.],
                                            [.6, 1., 1.],
                                            [.7, .5, 1.]],
                                           index=meth_stats.counts.index,
                                           columns=exp_columns_idx)
    assert_frame_equal(meth_stats.stats['beta-value'],
                       expected_beta_values_df)

def test_add_beta_zscore_stat(new_flat_meth_stats):
    meth_stats = MethStats.from_flat_dataframe(new_flat_meth_stats)
    meth_stats.add_beta_value_stat()
    meth_stats.add_zscores('beta-value')
    zscores = meth_stats.stats['beta-value_zscores']

    exp_columns_idx = (meth_stats.counts.columns
                       .drop(['n_meth', 'beta_value'], level=-1)
                       .droplevel(-1))
    assert_index_equal(exp_columns_idx, zscores.columns)

    assert_index_equal(meth_stats.counts.index, zscores.index)

    assert_frame_equal(_row_zscore(meth_stats.stats['beta-value']),
                       zscores)


def test_add_delta_meth_rep_level(new_flat_meth_stats):
    meth_stats = MethStats.from_flat_dataframe(new_flat_meth_stats)
    meth_stats.add_beta_value_stat()
    meth_stats.add_deltas(stat_name='beta-value', root=('hsc', '1'))
    meth_stats.add_deltas(stat_name='beta-value', root=('hsc', '1'),
                          output_name='beta-value_delta')
    assert_frame_equal(meth_stats.stats['beta-value_delta_hsc_1'],
                       meth_stats.stats['beta-value_delta'])
    # We test separately that the columns and index of the beta_value stats are correct
    meth_stats.stats['beta-value'].iloc[:, :] = np.array(
            [[0, 0, .5],
             [0, .4, .4],
             [0, -.2, .3]], dtype='f8')
    assert_frame_equal(meth_stats.stats['beta-value'],
                       meth_stats.stats['beta-value_delta'])


def test_add_delta_meth_pop_level(new_flat_meth_stats):
    meth_stats = (MethStats
                  .from_flat_dataframe(new_flat_meth_stats)
                  .convert_to_populations())
    meth_stats.add_beta_value_stat()
    meth_stats.add_deltas(stat_name='beta-value', root='hsc')
    meth_stats.add_deltas(stat_name='beta-value', root='hsc',
                          output_name='beta-value_delta')
    assert_frame_equal(meth_stats.stats['beta-value_delta_hsc'],
                       meth_stats.stats['beta-value_delta'])
    # We test separately that the columns and index of the beta_value stats are correct
    meth_stats.stats['beta-value'].iloc[:, :] = np.array(
            [[0, .5],
             [0, .2],
             [0, .4]], dtype='f8')
    assert_frame_equal(meth_stats.stats['beta-value'],
                       meth_stats.stats['beta-value_delta'])


def _row_zscore(df):
    return df.subtract(df.mean(axis=1), axis=0).divide(df.std(axis=1), axis=0)


@pytest.fixture()
def element_meth_stats():
    meth_stats_df = read_csv_with_padding("""\
Subject    ,       ,     , hsc        , hsc    , hsc     , hsc        , hsc    , hsc     , mpp1       , mpp1   , mpp1
Replicate  ,       ,     , 1          , 1      , 1       , 2          , 2      , 2       , 1          , 1      , 1
Stat       ,       ,     , beta_value , n_meth , n_total , beta_value , n_meth , n_total , beta_value , n_meth , n_total
Chromosome , Start , End ,            ,        ,         ,            ,        ,         ,            ,        ,
1          , 1     , 3   , 0.5        , 5      , 10      , 0.         , 0      , 10      , 1          , 10     , 10
1          , 4     , 6   , 1.         , 10     , 10      , 0.6        , 6      , 10      , 1          , 10     , 10
1          , 10    , 12  , 0.6        , 6      , 10      , 0.5        , 5      , 10      , 1          , 10     , 10
2          , 5     , 7   , 0.         , 0      , 10      , 1          , 10     , 10      , 0.         , 0      , 10
2          , 10    , 12  , 1          , 10     , 10      , 1          , 10     , 10      , 0.         , 0      , 10
    """, index_col=[0, 1, 2], header=[0, 1, 2])

    anno = pd.DataFrame({'region_id': [0, 0, 1, 2, 2]},
                        index=meth_stats_df.index)

    return meth_stats_df, anno

def test_inputs_are_not_modified(element_meth_stats):
    meth_stats_df, anno = element_meth_stats

    orig_meth_stats = meth_stats_df.copy(deep=True)
    orig_anno = anno.copy(deep=True)

    meth_stats = ml.MethStats(meth_stats=meth_stats_df,
                              anno=anno)
    assert_frame_equal(orig_anno, anno)
    assert_frame_equal(orig_meth_stats, meth_stats_df)

    meth_stats = ml.MethStats(element_meth_stats=meth_stats_df,
                              element_anno=anno)
    assert_frame_equal(orig_anno, anno)
    assert_frame_equal(orig_meth_stats, meth_stats_df)


def test_aggregate_element_meth_stats(element_meth_stats):
    meth_stats_df, element_anno = element_meth_stats

    meth_stats = ml.MethStats(element_meth_stats=meth_stats_df,
                              element_anno=element_anno)
    meth_stats.aggregate_element_counts()

    expected_counts_df = read_csv_with_padding("""\
Subject    ,       ,     ,           , hsc    , hsc     , hsc    , hsc     , mpp1   , mpp1
Replicate  ,       ,     ,           , 1      , 1       , 2      , 2       , 1      , 1
Stat       ,       ,     ,           , n_meth , n_total , n_meth , n_total , n_meth , n_total
Chromosome , Start , End , region_id ,        ,         ,        ,         ,        ,
1          , 1     , 6   , 0         , 15     , 20      , 6      , 20      , 20     , 20
1          , 10    , 12  , 1         , 6      , 10      , 5      , 10      , 10     , 10
2          , 5     , 12  , 2         , 10     , 20      , 20     , 20      , 0      , 20
    """, index_col=[0, 1, 2, 3], header=[0, 1, 2])

    # Process the expected counts df to make it compliant with the counts
    # contract
    expected_counts_df = ml.MethStats(meth_stats=expected_counts_df).counts

    assert_frame_equal(meth_stats.counts, expected_counts_df)

def test_save_flat(tmpdir, hierarchical_counts3, anno3, flat_meth_stats3):
    tmpdir = Path(tmpdir)
    meth_stats = ml.MethStats(meth_stats=hierarchical_counts3,
                              anno=anno3)

    # files can be saved in different formats
    output_tsv = str(tmpdir / 'test.tsv')
    output_p = str(tmpdir / 'test.p')
    meth_stats.save_flat([output_tsv, output_p])
    flat_meth_stats_from_p = pd.read_pickle(output_p)
    flat_meth_stats_from_tsv = pd.read_csv(
            output_tsv, sep='\t', header=0,
            dtype={'Chromosome': CategoricalDtype(categories=['1', '2'], ordered=True)})
    assert_frame_equal(flat_meth_stats_from_p, flat_meth_stats_from_tsv)
    expected_flat_meth_stats = flat_meth_stats3.rename(columns=ml.MethStats.grange_col_name_mapping).assign(
            Chromosome = lambda df: pd.Categorical(df['Chromosome'].astype(str), categories=['1', '2'],
                                                   ordered=True)
    )
    assert_frame_equal(flat_meth_stats_from_tsv, expected_flat_meth_stats)

    # check for bug which occured when the same meth stats object was saved twice
    meth_stats.save_flat([output_tsv])
    flat_meth_stats_from_tsv = pd.read_csv(
            output_tsv, sep='\t', header=0,
            dtype={'Chromosome': CategoricalDtype(categories=['1', '2'], ordered=True)})
    assert_frame_equal(flat_meth_stats_from_p, flat_meth_stats_from_tsv)




    # def calculate_region_interval_stats(pops: List[str], T: float, dmr_dfs: List[pd.DataFrame], metadata_table):
    #     print('RERUN')
    #     df = get_clustered_df(pops, dmr_dfs, metadata_table)
    #
    #     cpg_meth_stats = v1_bed_calls.intersect(df, n_cores=24)
    #     cpg_meth_stats_pop = cpg_meth_stats.convert_to_populations()
    #
    #     dmr_meth_stats_df = (cpg_meth_stats_pop
    #                          .df
    #                          .groupby(cpg_meth_stats_pop.anno.region_id)
    #                          .sum())
    #     dmr_meth_stats_df.index = df.set_index(ml.gr_names.all).index
    #     dmr_meth_stats = MethStats(meth_stats=dmr_meth_stats_df)
    #     dmr_meth_stats.add_beta_values()
    #
    #     cpg_beta_df = cpg_meth_stats_pop.df.loc[:, ncls(Stat='beta_value')]
    #     cpg_beta_df.columns = cpg_beta_df.columns.droplevel(1)
    #
    #     region_stats = dict()
    #
    #     cpg_delta_df = cpg_beta_df.subtract(cpg_beta_df['hsc'], axis=0)
    #     interval_auc = cpg_delta_df.groupby(cpg_meth_stats.anno['region_id']).sum()
    #     interval_auc.index = dmr_meth_stats.df.index
    #     interval_auc_norm = interval_auc.div(interval_auc.abs().max(axis=1), axis=0)
    #
    #     region_stats['auc'] = interval_auc
    #     region_stats['auc-norm'] = interval_auc_norm
    #     region_stats['beta-value'] = dmr_meth_stats.df.loc[:, ncls(Stat='beta_value')]
    #     region_stats['beta-value'].columns = region_stats['beta-value'].columns.droplevel(1)
    #     region_stats['beta-value_delta'] = region_stats['beta-value'].subtract(region_stats['beta-value']['hsc'], axis=0)
    #     region_stats['beta-value_z'] = zscore(region_stats['beta-value'])
    #
    #     delta_relevance_T = 0.1
    #     interval_n_relevant_bo_delta = cpg_delta_df.abs().gt(delta_relevance_T).groupby(cpg_meth_stats_pop.anno['region_id']).sum()
    #     interval_n_relevant_bo_delta.index = dmr_meth_stats.df.index
    #     region_stats['n-relevant_delta'] = interval_n_relevant_bo_delta
    #     region_stats['n-relevant_delta_norm']= interval_n_relevant_bo_delta.div(interval_n_relevant_bo_delta.max(axis=1), axis=0)
    #
    #     return region_stats

