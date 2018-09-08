from pathlib import Path

import pandas as pd
import numpy as np
import pytest

import methlevels as ml
from methlevels.utils import (
    read_csv_with_padding,
)

meth_calls_df = read_csv_with_padding("""
    Subject    ,       ,     ,           , hsc    , hsc     , mpp4   , mpp4    , b-cells , b-cells
    Statistic  ,       ,     ,           , n_meth , n_total , n_meth , n_total , n_meth  , n_total
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


@pytest.mark.parametrize('segmented', [False, True])
@pytest.mark.parametrize('custom_region_part_col_name', [False, True])
def test_region_plot(tmpdir, segmented, custom_region_part_col_name):

    if segmented:
        anno_df = pd.DataFrame({'region_id': 0,
                                'region_part': np.repeat('left_flank dmr dms right_flank'.split(), 2)},
                               index=meth_calls_df.index)
    else:
        anno_df = pd.DataFrame({'region_id': 0,
                                'region_part': np.repeat('left_flank dmr dmr right_flank'.split(), 2)},
                               index=meth_calls_df.index)

    tmpdir = Path(tmpdir)

    meth_stats = ml.MethStats(meth_calls_df, anno_df)
    meth_stats.add_beta_values()

    tidy_anno = (meth_stats.anno
                 .drop(list(set(meth_stats.anno.index.names) & set(meth_stats.anno.columns)), axis=1)
                 .reset_index())

    tidy_meth_stats = meth_stats.df.stack(0).reset_index()
    tidy_meth_stats.columns.name = None

    if custom_region_part_col_name:
        tidy_anno = tidy_anno.rename(columns={'region_part': 'custom_region_part'})
        region_plot = ml.RegionPlot(tidy_meth_stats, tidy_anno, region_part_col='custom_region_part')
    else:
        region_plot = ml.RegionPlot(tidy_meth_stats, tidy_anno)

    fig = region_plot.grid_plot('mpp4',
                                bp_width_100=8, row_height=1, bp_padding=10)
    fig.savefig(tmpdir / 'test.png')
    fig = region_plot.grid_plot('mpp4', highlighted_subjects=['mpp4'],
                                bp_width_100=8, row_height=1, bp_padding=10)
    fig.savefig(tmpdir / 'test.png')
    fig = region_plot.grid_plot('mpp4', highlighted_subjects=['hsc', 'mpp4'],
                                bp_width_100=8, row_height=1, bp_padding=10)
    fig.savefig(tmpdir / 'test.png')
