import numpy as np
import pandas as pd
from dpcontracts import invariant

from methlevels import gr_names


def _dmr_intervals_data_contract(self):
    """DMR Interval data contract

    Additional annotation columns are allowed

    - always have region id, self.region_id_col
    """
    assert list(self.df.columns[0:3]) == gr_names.all
    assert self.df[gr_names.chrom].dtype.name == 'category'
    assert isinstance(self.df[gr_names.chrom].cat.categories[0], str)
    assert self.df[gr_names.start].dtype == self.df[gr_names.end].dtype == np.int64
    assert self.region_id_col in self.df.columns
    assert self.region_part_col in self.df.columns
    assert isinstance(self.df.index, pd.RangeIndex)

    # sorted by region id, then by chr, start, end
    assert self.df[self.region_id_col].is_monotonic
    assert self.df[gr_names.chrom].is_monotonic
    assert (self.df.groupby(self.region_id_col)[gr_names.start]
            .agg(lambda ser: ser.is_monotonic).all())
    assert (self.df.groupby(self.region_id_col)[gr_names.end]
            .agg(lambda ser: ser.is_monotonic).all())
    return True

@invariant('correct format', _dmr_intervals_data_contract)
class DMRIntervals:
    """DMR dataframe with annotations"""

    def __init__(self, df: pd.DataFrame,
                 region_id_col: str = 'region_id',
                 region_part_col: str = 'region_part') -> None:
        self.df = df
        self.region_id_col = region_id_col
        self.region_part_col = region_part_col

    def add_flanks(self, n_bp: int):
        """region id will remain in index, plus region_part

        anno cols after coord cols
        region id and region part also as cols at the end (and alos as index)
        """

        def add_flanks(group_df):
            left_flank_df = group_df.iloc[[0]].copy()
            left_flank_df['End'] = left_flank_df['Start']
            left_flank_df['Start'] -= n_bp
            left_flank_df[self.region_part_col] = 'left_flank'
            right_flank_df = group_df.iloc[[-1]].copy()
            right_flank_df['Start'] = right_flank_df['End']
            right_flank_df['End'] += n_bp
            right_flank_df[self.region_part_col] = 'right_flank'
            return pd.concat([left_flank_df, group_df, right_flank_df], axis=0)

        self.df = (self.df
                   .groupby(self.region_id_col, group_keys=False)
                   .apply(add_flanks)
                   .reset_index(drop=True)
                   # .assign(running_idx = lambda df: np.arange(df.shape[0]))
                   # .set_index([self.region_id_col, 'running_idx'], drop=False, append=False)
                   # .drop('running_idx', axis=1)
                   )

        return self
