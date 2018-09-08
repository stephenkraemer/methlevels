from copy import deepcopy
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from methlevels import MethStats
from methlevels.utils import read_csv_with_padding

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
def test_convert_to_populations(flat_meth_stats, with_anno_cols):
    if with_anno_cols:
        flat_meth_stats['size'] = 3
    pop_level_meth_stats = MethStats.from_flat_dataframe(flat_meth_stats).convert_to_populations()
    assert pop_level_meth_stats.df.loc[('1', 3, 4), ('hsc', 'n_meth')] == 25
    dtypes=pop_level_meth_stats.df.dtypes
    assert dtypes.loc[nidxs(Stat='beta_value')].eq(np.float).all()
    assert dtypes.loc[nidxs(Stat='n_meth')].eq(np.int).all()
    assert dtypes.loc[nidxs(Stat='n_total')].eq(np.int).all()
    if with_anno_cols:
        assert pop_level_meth_stats.anno['size'].eq(3).all()


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


def test_additional_index_column(flat_meth_stats):

    flat_meth_stats = pd.concat([flat_meth_stats, flat_meth_stats])

    flat_meth_stats['anno1'] = list('abcabc')
    flat_meth_stats['region_id'] = [0, 0, 0 , 1, 1, 1]
    meth_stats = MethStats.from_flat_dataframe(flat_meth_stats,
                                               additional_index_cols=['region_id'])

    expected_df = read_csv_with_padding("""\
        Subject    ,       ,     ,           , hsc        , hsc    , hsc     , hsc        , hsc    , hsc     , mpp1       , mpp1   , mpp1
        Replicate  ,       ,     ,           , 1          , 1      , 1       , 2          , 2      , 2       , 1          , 1      , 1
        Stat       ,       ,     ,           , beta_value , n_meth , n_total , beta_value , n_meth , n_total , beta_value , n_meth , n_total
        Chromosome , Start , End , region_id ,            ,        ,         ,            ,        ,         ,            ,        ,
        1          , 1     , 2   , 0         , 0.5        , 10     , 10      , 0.5        , 10     , 10      , 0.5        , 10     , 10
        1          , 1     , 2   , 1         , 0.5        , 10     , 10      , 0.5        , 10     , 10      , 0.5        , 10     , 10
        1          , 3     , 4   , 0         , 0.5        , 10     , 10      , 0.5        , 15     , 10      , 0.5        , 10     , 10
        1          , 3     , 4   , 1         , 0.5        , 10     , 10      , 0.5        , 15     , 10      , 0.5        , 10     , 10
        2          , 5     , 6   , 0         , 0.5        , 10     , 10      , 0.5        , 10     , 10      , 0.5        , 10     , 10
        2          , 5     , 6   , 1         , 0.5        , 10     , 10      , 0.5        , 10     , 10      , 0.5        , 10     , 10
    """, index_col=[0, 1, 2, 3], header=[0, 1, 2])

    expected_anno = read_csv_with_padding("""\
        Chromosome , Start , End , region_id , anno1
        1          , 1     , 2   , 0         , a
        1          , 1     , 2   , 1         , a
        1          , 3     , 4   , 0         , b
        1          , 3     , 4   , 1         , b
        2          , 5     , 6   , 0         , c
        2          , 5     , 6   , 1         , c
    """, index_col=[0, 1, 2, 3], header=[0])

    meth_stats_expected = MethStats(meth_stats=expected_df, anno=expected_anno)

    assert_frame_equal(meth_stats.df, meth_stats_expected.df, check_names=False)
    assert_frame_equal(meth_stats.anno, meth_stats_expected.anno, check_names=False)

    meth_stats = MethStats.from_flat_dataframe(flat_meth_stats,
                                               additional_index_cols=['region_id'],
                                               drop_additional_index_cols=False)

    expected_anno_no_drop = read_csv_with_padding("""\
        Chromosome , Start , End , region_id , anno1 , region_id
        1          , 1     , 2   , 0         , a     , 0
        1          , 1     , 2   , 1         , a     , 1
        1          , 3     , 4   , 0         , b     , 0
        1          , 3     , 4   , 1         , b     , 1
        2          , 5     , 6   , 0         , c     , 0
        2          , 5     , 6   , 1         , c     , 1
    """, index_col=[0, 1, 2, 3], header=[0]).rename(columns={'region_id.1': 'region_id'})
    print(expected_anno_no_drop)

    meth_stats_expected_no_drop = MethStats(meth_stats=expected_df, anno=expected_anno_no_drop)
    #
    # assert_frame_equal(meth_stats.df, meth_stats_expected_no_drop.df, check_names=False)
    # assert_frame_equal(meth_stats.anno, meth_stats_expected_no_drop.anno, check_names=False)
