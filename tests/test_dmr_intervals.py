import pandas as pd
from methlevels import DMRIntervals, MethStats

def test_add_flanks():

    interval_df = pd.DataFrame([
        ['1', 10, 20],
        ['2', 10, 20],
    ], columns=MethStats.grange_col_names)
    interval_df.index.name = 'region_id'
    interval_df['anno'] = [1, 2]


    DMRIntervals(interval_df).add_flanks(2000).df

