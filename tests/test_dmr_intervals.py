import numpy as np
from pandas.testing import assert_frame_equal
from pandas.api.types import CategoricalDtype
from methlevels import DMRIntervals
from methlevels.utils import read_csv_with_padding

gr_dtypes_d = {
    'Chromosome': CategoricalDtype(categories=['1', '2'], ordered=True),
    'Start': np.int64,
    'End': np.int64,

}

def test_add_flanks_plain_dmrs():

    interval_df = read_csv_with_padding("""\
Chromosome , Start , End   , anno , custom_region_id_name , custom_region_part_name
1          , 10000 , 20000 , a    , 1                     , dmr
2          , 10000 , 20000 , b    , 2                     , dmr
    """, dtype=gr_dtypes_d)

    dmr_intervals = DMRIntervals(interval_df,
                                 region_id_col='custom_region_id_name',
                                 region_part_col='custom_region_part_name')
    dmr_intervals = dmr_intervals.add_flanks(2000)

    expected_df = read_csv_with_padding("""\
Chromosome , Start , End   , anno , custom_region_id_name , custom_region_part_name
1          , 8000  , 10000 , a    , 1                     , left_flank
1          , 10000 , 20000 , a    , 1                     , dmr
1          , 20000 , 22000 , a    , 1                     , right_flank
2          , 8000  , 10000 , b    , 2                     , left_flank
2          , 10000 , 20000 , b    , 2                     , dmr
2          , 20000 , 22000 , b    , 2                     , right_flank
    """, dtype=gr_dtypes_d)

    assert_frame_equal(dmr_intervals.df, expected_df)

def test_add_flanks_segmented_dmrs():

    interval_df = read_csv_with_padding("""\
        Chromosome , Start , End   , anno , custom_region_id_name , custom_region_part_name
        1          , 10000 , 13000 , a    , 1                     , dms1
        1          , 13000 , 17000 , a    , 1                     , dms2
        1          , 17000 , 20000 , a    , 1                     , dms3
        1          , 15000 , 20000 , b    , 2                     , dms1
        1          , 20000 , 25000 , b    , 2                     , dms2
        2          , 10000 , 15000 , b    , 3                     , dms1
        2          , 15000 , 20000 , b    , 3                     , dms2
    """, dtype=gr_dtypes_d)

    expected_flanked_df = read_csv_with_padding("""\
        Chromosome , Start , End   , anno , custom_region_id_name , custom_region_part_name
        1          , 8000  , 10000 , a    , 1                     , left_flank
        1          , 10000 , 13000 , a    , 1                     , dms1
        1          , 13000 , 17000 , a    , 1                     , dms2
        1          , 17000 , 20000 , a    , 1                     , dms3
        1          , 20000 , 22000 , a    , 1                     , right_flank
        1          , 13000  , 15000 , b    , 2                     , left_flank
        1          , 15000 , 20000 , b    , 2                     , dms1
        1          , 20000 , 25000 , b    , 2                     , dms2
        1          , 25000 , 27000 , b    , 2                     , right_flank
        2          , 8000  , 10000 , b    , 3                     , left_flank
        2          , 10000 , 15000 , b    , 3                     , dms1
        2          , 15000 , 20000 , b    , 3                     , dms2
        2          , 20000 , 22000 , b    , 3                     , right_flank
        """, dtype=gr_dtypes_d)

    dmr_intervals = DMRIntervals(interval_df,
                                 region_id_col='custom_region_id_name',
                                 region_part_col='custom_region_part_name')
    dmr_intervals = dmr_intervals.add_flanks(2000)

    assert_frame_equal(dmr_intervals.df, expected_flanked_df)
