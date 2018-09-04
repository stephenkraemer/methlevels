#-
import tempfile
from pathlib import Path
from joblib import Parallel, delayed
from io import StringIO
import re
import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.api.types import CategoricalDtype

import pandas as pd
import numpy as np

import methlevels as ml
from methlevels.utils import read_csv_with_padding
import subprocess
import pytest
#-




tmpdir_obj = tempfile.TemporaryDirectory()
tmpdir_str = tmpdir_obj.name
tmpdir_path = Path(tmpdir_obj.name)


meth_calls_bed_df = read_csv_with_padding("""\
#chrom , start , end , n_meth , n_total , beta_value
1      , 10    , 12  , 1      , 12      , 0.0833333333
1      , 20    , 22  , 2      , 12      , 0.1666666667
1      , 30    , 32  , 3      , 12      , 0.25
1      , 40    , 42  , 4      , 12      , 0.3333333333
1      , 50    , 52  , 5      , 12      , 0.4166666667
1      , 52    , 54  , 5      , 12      , 0.4166666667
1      , 54    , 56  , 5      , 12      , 0.4166666667
1      , 60    , 62  , 6      , 12      , 0.5
1      , 70    , 72  , 7      , 12      , 0.5833333333
1      , 80    , 82  , 8      , 12      , 0.6666666667
1      , 110   , 112 , 9      , 12      , 0.75
1      , 120   , 122 , 10     , 12      , 0.8333333333
1      , 130   , 132 , 11     , 12      , 0.9166666667
1      , 140   , 142 , 12     , 12      , 1
2      , 10    , 12  , 1      , 4       , 0.25
2      , 20    , 22  , 2      , 4       , 0.5
2      , 30    , 32  , 3      , 4       , 0.75
2      , 40    , 42  , 4      , 4       , 1
    """)

chrom_dtype = CategoricalDtype(['1', '2'], True)

meth_calls_bed_df.to_csv(tmpdir_path / 'subject1.bed', sep='\t', index=False)
meth_calls_bed_df.to_csv(tmpdir_path / 'subject2.bed', sep='\t', index=False)

subprocess.run(['bgzip', tmpdir_path / 'subject1.bed'], check=True)
subprocess.run(['bgzip', tmpdir_path / 'subject2.bed'], check=True)

subprocess.run(['tabix', '-p', 'bed', tmpdir_path / 'subject1.bed.gz'], check=True)
subprocess.run(['tabix', '-p', 'bed', tmpdir_path / 'subject2.bed.gz'], check=True)


metadata_table = pd.DataFrame({
    'bed_path': [tmpdir_path / 'subject1.bed.gz', tmpdir_path / 'subject2.bed.gz'],
    'Subject': ['subject1', 'subject2'],
    'sample_id': ['subject1', 'subject2'],
})

bed_calls = ml.BedCalls(metadata_table=metadata_table, tmpdir=tmpdir_str,
                        pop_order=metadata_table['Subject'].values,
                        n_meth_col=3, n_total_col=4, beta_value_col=5)

intervals_df = pd.DataFrame({
    'Chromosome': ['1', '1', '1', '2'],
    'Start': [31, 40, 120, 20],
    'End':   [60, 70, 131, 30],
    'Anno1': list('abcd'),
    'Region ID': [0, 1, 2, 3],
})
intervals_df.index.name = 'region_id'


def test_aggregate(tmpdir):

    expected_df = ml.MethStats(ml.utils.read_csv_with_padding("""\
        Subject    ,       ,     , subject1     , subject1 , subject1 , subject2     , subject2 , subject2
        Stat       ,       ,     , beta_value   , n_meth   , n_total  , beta_value   , n_meth   , n_total
        Chromosome , Start , End ,              ,          ,          ,              ,          ,
        1          , 31    , 60  , 0.3666666664 , 22       , 60       , 0.3666666664 , 22       , 60
        1          , 40    , 70  , 0.4166666667 , 25       , 60       , 0.4166666667 , 25       , 60
        1          , 120   , 131 , 0.875        , 21       , 24       , 0.875        , 21       , 24
        2          , 20    , 30  , 0.5          , 2        , 4        , 0.5          , 2        , 4
        """, header=[0, 1], index_col=[0, 1, 2])).df

    meth_stats = bed_calls.aggregate(intervals_df, n_cores=2)

    assert_frame_equal(meth_stats.df, expected_df)


@pytest.mark.parametrize('worker_pool', [False, True])
def test_intersect(tmpdir, worker_pool):

    def run_intersect(parallel):
        meth_stats = bed_calls.intersect(intervals_df, n_cores=2,
                                         additional_index_cols=['Region ID'], parallel=parallel)

        expected_df = ml.MethStats(read_csv_with_padding("""\
            Subject    ,       ,     ,           , subject1     , subject1 , subject1 , subject2     , subject2 , subject2
            Stat       ,       ,     ,           , beta_value   , n_meth   , n_total  , beta_value   , n_meth   , n_total
            Chromosome , Start , End , Region ID ,              ,          ,          ,              ,          ,
            1          , 30    , 32  , 0         , 0.25         , 3        , 12       , 0.25         , 3        , 12
            1          , 40    , 42  , 0         , 0.3333333333 , 4        , 12       , 0.3333333333 , 4        , 12
            1          , 40    , 42  , 1         , 0.3333333333 , 4        , 12       , 0.3333333333 , 4        , 12
            1          , 50    , 52  , 0         , 0.4166666667 , 5        , 12       , 0.4166666667 , 5        , 12
            1          , 50    , 52  , 1         , 0.4166666667 , 5        , 12       , 0.4166666667 , 5        , 12
            1          , 52    , 54  , 0         , 0.4166666667 , 5        , 12       , 0.4166666667 , 5        , 12
            1          , 52    , 54  , 1         , 0.4166666667 , 5        , 12       , 0.4166666667 , 5        , 12
            1          , 54    , 56  , 0         , 0.4166666667 , 5        , 12       , 0.4166666667 , 5        , 12
            1          , 54    , 56  , 1         , 0.4166666667 , 5        , 12       , 0.4166666667 , 5        , 12
            1          , 60    , 62  , 1         , 0.5          , 6        , 12       , 0.5          , 6        , 12
            1          , 120   , 122 , 2         , 0.8333333333 , 10       , 12       , 0.8333333333 , 10       , 12
            1          , 130   , 132 , 2         , 0.9166666667 , 11       , 12       , 0.9166666667 , 11       , 12
            2          , 20    , 22  , 3         , 0.5          , 2        , 4        , 0.5          , 2        , 4
            """, header=[0, 1], index_col=[0, 1, 2, 3])).df

        assert_frame_equal(meth_stats.df, expected_df)

        expected_anno = pd.DataFrame({'Anno1': list('aababababbccd')}, index=expected_df.index)
        expected_anno['Region start'] = expected_anno['Anno1'].map({'a': 31, 'b': 40, 'c': 120, 'd': 20})
        expected_anno['Region end'] = expected_anno['Anno1'].map({'a': 60, 'b': 70, 'c': 131, 'd': 30})
        expected_anno = expected_anno[['Region start', 'Region end', 'Anno1']]

        assert_frame_equal(meth_stats.anno, expected_anno)

    if worker_pool:
        with Parallel(2) as parallel:
            run_intersect(parallel)
    else:
        run_intersect(None)


