from pathlib import Path

import pandas as pd
import numpy as np
import pytest

import methlevels as ml
from methlevels.utils import (
    read_csv_with_padding,
)
from methlevels.utils import NamedColumnsSlice as ncls

@pytest.fixture()
def meth_calls_df():
    meth_calls_df = read_csv_with_padding("""
        Subject    ,       ,     ,           , hsc    , hsc     , mpp4   , mpp4    , b-cells , b-cells
        Stat  ,       ,     ,           , n_meth , n_total , n_meth , n_total , n_meth  , n_total
        Chromosome , Start , End , region_id ,        ,         ,        ,         ,         ,
        1          , 10    , 12  , 0         , 10      , 10      , 30      , 30      , 20      , 20
        1          , 20    , 22  , 0         , 20      , 20      , 30      , 30      , 20      , 20
        1          , 22    , 24  , 0         , 30      , 30      , 10      , 20      , 0      , 20
        1          , 24    , 26  , 0         , 30      , 30      , 10      , 20      , 0      , 20
        1          , 30    , 32  , 0         , 30      , 30      , 20      , 20      , 10      , 20
        1          , 34    , 36  , 0         , 30      , 30      , 20      , 20      , 10      , 20
        1          , 44    , 46  , 0         , 30      , 20      , 10      , 10      , 20      , 20
        1          , 50    , 52  , 0         , 10      , 10      , 10      , 10      , 20      , 20
    """, header=[0, 1], index_col=[0, 1, 2, 3])
    return meth_calls_df


@pytest.mark.parametrize('segmented', [False, True])
@pytest.mark.parametrize('custom_region_part_col_name', [False, True])
@pytest.mark.parametrize('multiple_regions', [False, True])
def test_region_plot(tmpdir, meth_calls_df, segmented,
                     custom_region_part_col_name, multiple_regions):

    tmpdir = Path(tmpdir)

    if custom_region_part_col_name:
        region_id_col = 'custom_region_id'
        region_part_col = 'custom_region_part'
    else:
        region_part_col='region_part'
        region_id_col = 'region_id'

    if segmented:
        anno_df = pd.DataFrame({region_id_col: 0,
                                region_part_col: np.repeat('left_flank dmr dms right_flank'.split(), 2)},
                               index=meth_calls_df.index)
    else:
        anno_df = pd.DataFrame({region_id_col: 0,
                                region_part_col: np.repeat('left_flank dmr dmr right_flank'.split(), 2)},
                               index=meth_calls_df.index)

    if multiple_regions:
        anno_df2 = anno_df.copy()
        anno_df2[region_id_col] = 1
        anno_df2.rename(index={0: 1}, inplace=True)
        anno_df = pd.concat([anno_df, anno_df2]).sort_index()
        meth_calls_df2 = meth_calls_df.copy()
        meth_calls_df2.rename({0: 1}, inplace=True)
        meth_calls_df2.loc[:, ncls(Stat='n_meth')] = (
                meth_calls_df2.loc[:, ncls(Stat='n_meth')] / 2)
        meth_calls_df = pd.concat([meth_calls_df, meth_calls_df2]).sort_index()

    meth_stats = ml.MethStats(meth_calls_df, anno_df)

    if custom_region_part_col_name:
        region_plot = ml.DMRPlot(meth_stats,
                                 region_part_col=region_part_col,
                                 region_id_col = region_id_col)
    else:
        region_plot = ml.DMRPlot(meth_stats)

    for highlighted_subjects in [None, ['mpp4'], ['hsc', 'mpp4']]:
        region_plot.grid_plot(tmpdir, 'mpp4-dmrs',
                              highlighted_subjects=highlighted_subjects,
                              bp_width_100=8, row_height=1, bp_padding=10)
