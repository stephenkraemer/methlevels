#-
import tempfile
import time
from io import StringIO
from itertools import chain
import joblib
from joblib import Parallel, delayed
from methlevels import MethStats
from pathlib import Path
from subprocess import run, PIPE
from typing import List, Optional

import pandas as pd
idxs = pd.IndexSlice
import pyranges as pr

from methlevels.utils import has_duplicated_coord
pd.DataFrame.has_duplicated_coord = has_duplicated_coord
#-


def _run_tabix(bed_path, intervals_bed_fp, bed_calls):
    process = run(['tabix', '-R', intervals_bed_fp, bed_path], stdout=PIPE, encoding='utf-8',
                  check=True)
    return pd.read_csv(StringIO(process.stdout), sep='\t', header=None,
                       usecols=bed_calls.stat_col_index_input,
                       names=bed_calls.stat_col_order_input,
                       dtype=bed_calls.stat_col_dtypes)[bed_calls.stat_col_order_output]

def _run_tabix_agg(bed_path, intervals_bed_fp, groupby_seq, bed_calls, int_index):
    # avoid aligning on index
    assert not isinstance(groupby_seq, (pd.DataFrame, pd.Series))
    per_cpg_df = _run_tabix(bed_path, intervals_bed_fp, bed_calls).iloc[int_index, :]
    per_interval_df = (per_cpg_df
                       .groupby(groupby_seq)
                       .sum()
                       .assign(beta_value = lambda df: df['n_meth'] / df['n_total']))
    return per_interval_df

#-

class BedCalls:
    """Interface to a set of methylation calls in BED format

    Flexible and fast interactive queries of BED files.

    Args:
        metadata_table: Available BED files with annotations
            Required columns (May have arbitrary other columns)
            - bed path: absolute path to tabixed bed.gz file
            - subject: name of the biological entity. The same entity may be present in
                multiple replicates
            - sample_id: unique sample id
        n_meth_col, n_total_col, beta_value_col: 0-based column indices.


    Prerequisites:
        - environment: htslib (tabix)
        - BED files must be bgzipped and indexed with tabix


    Required BED format:


    Several query methods take an interval dataframe with format:
        Required cols (must be the first three columns):
        - Chromosome
        - Start
        - End

        Intervals are specified according to BED convention. Partial overlaps
        are counted

        Optional cols:
        - any other column will be treated as annotation column and added
          to the result

        Index:
        - should be simple integer index, will be ignored

    """

    grange_col_names = 'Chromosome Start End'.split()
    stat_col_dtypes = {'beta_value': 'f8', 'n_meth': 'i8', 'n_total': 'i8'}
    stat_col_order_output = 'beta_value n_meth n_total'.split()

    def __init__(self, metadata_table: pd.DataFrame, tmpdir: str,
                 pop_order: List[str], beta_value_col: int,
                 n_meth_col: int, n_total_col: int) -> None:
        self.metadata_table = metadata_table
        self.tmpdir = tmpdir
        self.pop_order = pop_order
        self.beta_value_col = beta_value_col
        self.n_meth_col = n_meth_col
        self.n_total_col = n_total_col

        self.stat_col_index_input = [self.beta_value_col, self.n_meth_col, self.n_total_col]
        self.stat_col_order_input = [t[0] for t in sorted(
                zip(self.stat_col_order_output, self.stat_col_index_input), key=lambda t: t[1])]

    def aggregate(self, intervals_df: pd.DataFrame, n_cores: int, subjects=None) -> MethStats:

        if subjects:
            pop_order = [p for p in self.pop_order if p in subjects]
            metadata_table = self.metadata_table.query('Subject in @subjects')
        else:
            pop_order = self.pop_order
            metadata_table = self.metadata_table

        assert intervals_df.columns[0] in ['chr', '#chr', 'chromosome', 'Chromosome', 'chrom']
        assert intervals_df.columns[1:3].to_series().str.lower().tolist() == ['start', 'end']

        intervals_df = intervals_df.copy()
        intervals_df = intervals_df.rename(columns=MethStats.grange_col_name_mapping)
        intervals_df['uid'] = range(intervals_df.shape[0])
        intervals_gr = pr.PyRanges(intervals_df)
        intervals_gr_clustered = intervals_gr.cluster()

        with tempfile.TemporaryDirectory(dir=self.tmpdir) as curr_tmpdir:  #type: ignore

            intervals_bed_fp = Path(curr_tmpdir).joinpath('intervals.bed')
            intervals_gr_clustered.df.iloc[:, 0:3].to_csv(intervals_bed_fp, sep='\t', header=False, index=False)

            index_df = self._get_index_df(intervals_bed_fp)
            index_df['int_index'] = range(index_df.shape[0])
            assert not index_df.has_duplicated_coord()

            print('WARNING: I have hardcoded strandedness to False for now')
            cpg_index_gr = pr.PyRanges(index_df).join(intervals_gr, strandedness=False)

            call_dfs = Parallel(n_cores)(
                    delayed(_run_tabix_agg)(bed_path,
                                            intervals_bed_fp=intervals_bed_fp,
                                            groupby_seq=cpg_index_gr.df['uid'].values,
                                            int_index=cpg_index_gr.df['int_index'],
                                            bed_calls=self)
                    for bed_path in metadata_table['bed_path'])

        calls_merged = pd.concat([intervals_df] + call_dfs, axis=1,
                                 keys=chain(['region'], metadata_table['sample_id']))

        new_idx = ['_'.join(t) for t in calls_merged.columns]
        new_idx[0:3] = self.grange_col_names
        calls_merged.columns = new_idx

        calls_merged = calls_merged.rename(columns={'Chromosome': 'chr', 'Start': 'start', 'End': 'end'})
        meth_stats = MethStats.from_flat_dataframe(calls_merged, pop_order=pop_order)

        return meth_stats


    def intersect(self, intervals_df: pd.DataFrame, n_cores: int = None,
                  additional_index_cols: Optional[List[str]] = None,
                  elements: bool = False,
                  drop_additional_index_cols=True, parallel: joblib.Parallel = None) -> MethStats:
        """Retrieve individual motifs, annotated with their parent interval
        
        Args:
            n_cores: ignored if parallel is specified
            additional_index_cols: passed on the MethStats.from_flat_dataframe
            drop_additional_index_cols: passed on the MethStats.from_flat_dataframe

        Optionally, annotation columns may be added to the intervals df.
        They will be added to the results (available through anno df of
        the returned MethStats object).

        Chromosome categories are taken from interval df, with order

        If the interval df contains duplicate genomic intervals, it must contain
        one or more additional index variables, so that the complete index
        if fully unique. These additional index variables must be distinguished
        from annotation variables by passing them via /additional_index_cols/

        """

        print('Deprecation warning: elements will be changed to True in the future')

        print('Prepare intervals')
        t1 = time.time()
        assert intervals_df.columns[0] in ['chr', '#chr', 'chromosome', 'Chromosome', 'chrom']
        assert intervals_df.columns[1:3].to_series().str.lower().tolist() == ['start', 'end']
        intervals_df.rename(columns=MethStats.grange_col_name_mapping)
        intervals_gr_unclustered = pr.PyRanges(intervals_df)
        intervals_gr_clustered = intervals_gr_unclustered
        # TODO reactivate this
        # intervals_gr_clustered = intervals_gr_unclustered.cluster()

        print('Done', time.time() - t1)

        print('Retrieve data from file system')
        t1 = time.time()
        with tempfile.TemporaryDirectory(dir=self.tmpdir) as curr_tmpdir:  # type: ignore
            intervals_bed_fp = Path(curr_tmpdir).joinpath('intervals.bed')
            intervals_gr_clustered.df.iloc[:, 0:3].to_csv(intervals_bed_fp, sep='\t', header=False, index=False)

            # Tabix will report duplicates if the input regions overlap
            index_df = self._get_index_df(intervals_bed_fp)
            assert not index_df.has_duplicated_coord()

            if parallel is None:
                parallel = Parallel(n_cores)
            call_dfs = parallel(delayed(_run_tabix)(bed_path, intervals_bed_fp=intervals_bed_fp, bed_calls=self)
                                for bed_path in self.metadata_table['bed_path'])
        print('Done', time.time() - t1)

        print('Merge data')
        t1 = time.time()
        calls_merged = pd.concat(call_dfs, axis=1, keys=self.metadata_table['sample_id'])
        calls_merged.columns = ['_'.join(t) for t in calls_merged.columns]
        calls_merged = pd.concat([index_df, calls_merged], axis=1)
        calls_merged_gr = pr.PyRanges(calls_merged)  # unordered categorical
        print('Done', time.time() - t1)

        print('Create flat dataframe')
        t1 = time.time()
        print('WARNING: I have hard-coded strandedness to False for now')
        annotated = calls_merged_gr.join(intervals_gr_unclustered, strandedness=False)
        flat_df = annotated.df.rename(columns={'Start_b': 'Region start', 'End_b': 'Region end'})
        # pyranges currently returns unordered categorical
        flat_df['Chromosome'] = flat_df['Chromosome'].cat.set_categories(
                index_df['Chromosome'].unique(), ordered=True)
        print('Done', time.time() - t1)

        print('Create methlevels')
        t1 = time.time()
        meth_levels = MethStats.from_flat_dataframe(flat_df, pop_order=self.pop_order,
                                                    elements=elements,
                                                    additional_index_cols=additional_index_cols,
                                                    drop_additional_index_cols=drop_additional_index_cols)
        print('Done', time.time() - t1)

        return meth_levels

    def _get_index_df(self, intervals_bed_fp):
        process = run(['tabix', '-R', intervals_bed_fp, self.metadata_table['bed_path'].iloc[0]],
                      stdout=PIPE, encoding='utf-8')
        # PyRanges has problems with categorical input
        index_df = pd.read_csv(StringIO(process.stdout), sep='\t', header=None, usecols=[0, 1, 2],
                               names='Chromosome Start End'.split(),
                               dtype={'Chromosome': str, 'Start': 'i8', 'End': 'i8'})
        return index_df



