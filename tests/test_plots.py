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
    1          , 10    , 12  , 0         , 2      , 10      , 8      , 30      , 18      , 20
    1          , 20    , 22  , 0         , 4      , 20      , 9      , 30      , 18      , 20
    1          , 22    , 24  , 0         , 6      , 30      , 5      , 20      , 18      , 20
    1          , 24    , 26  , 0         , 7      , 30      , 4      , 20      , 18      , 20
    1          , 30    , 32  , 0         , 3      , 20      , 2      , 10      , 18      , 20
    1          , 50    , 52  , 0         , 1      , 10      , 5      , 10      , 18      , 20
""", header=[0, 1], index_col=[0, 1, 2, 3])

anno_df = pd.DataFrame({'region_id': 0,
                        'region_part': np.repeat('left_flank dmr right_flank'.split(), 2)},
                       index=meth_calls_df.index)

def test_region_plot():
    # __file__ = '/home/kraemers/projects/methlevels/tests/test_plots.py'

    meth_stats = ml.MethStats(meth_calls_df, anno_df)
    meth_stats.add_beta_values()

    region_plot = ml.RegionPlot(meth_stats.df, meth_stats.anno)
    fig = region_plot.grid_plot('mpp4',
                                bp_width_100=8, row_height=1, bp_padding=10)
    fig = region_plot.grid_plot('mpp4', highlighted_subjects=['mpp4'],
                                bp_width_100=8, row_height=1, bp_padding=10)
    fig = region_plot.grid_plot('mpp4', highlighted_subjects=['hsc', 'mpp4'],
                                bp_width_100=8, row_height=1, bp_padding=10)
