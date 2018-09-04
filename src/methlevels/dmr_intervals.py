import pandas as pd
from pandas.api.types import CategoricalDtype
from dataclasses import dataclass, field

from methlevels import MethStats

@dataclass()
class DMRIntervals:
    """DMR dataframe and granges with annotations

    index is scalar

    intervals df can have annotation cols, must have index with region id



    """

    df: pd.DataFrame

    region_part_dtype: CategoricalDtype = field(
            default=CategoricalDtype(['left_flank', 'dmr', 'dms1', 'dms2', 'right_flank'], True),
            init=False,
            repr=False)
    coord_dtype: CategoricalDtype = field(
            default=CategoricalDtype(MethStats.grange_col_names, True),
            init=False,
            repr=False)


    def __post_init__(self):
        assert isinstance(self.df.index, pd.Index)
        assert self.df.index.name == 'region_id'

    def add_flanks(self, n_bp):
        """region id will remain in index, plus region_part

        anno cols after coord cols
        region id and region part also as cols at the end (and alos as index)
        """

        df = self.df

        df_with_flanks = pd.concat((df.assign(Start = df.Start - n_bp, End = df.Start),
                                 df,
                                 df.assign(Start = df.End, End = df.End + n_bp),
                                 ), keys=['left_flank', 'dmr', 'right_flank'], axis=1)

        df_cms = df_with_flanks.columns
        df_with_flanks.columns = pd.MultiIndex.from_arrays((
            df_cms.get_level_values(0).astype(self.region_part_dtype),
            df_cms.get_level_values(1).astype(str)
        ), names=['region_part', 'coord'])

        df = df_with_flanks.stack(0).loc[:, df.columns]
        df = pd.concat((df, df.index.to_frame()), axis=1)

        self.df = df

        return self

