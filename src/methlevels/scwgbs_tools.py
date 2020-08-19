from typing import Dict, Iterable, Union, List, Tuple

import pandas as pd
from joblib import Parallel, delayed

gr_cols = ["Chromosome", "Start", "End"]


def extract_and_concat_mcalls(
    mcall_df_fps: Union[List[str], pd.Series],
    sample_names: Union[List[str], pd.Series],
    cpg_gr_df: pd.DataFrame,
    n_cores: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Retrieve calls for a set of CpGs from individual mcall results

    Missing calls are filled with np.nan, which usually leads to casting n_meth and n_total
    to float.
    The row order and index of cpg_gr_df is maintained

    Parameters
    ----------
    mcall_df_fps
        paths to pickled meth calls.
        Calls are assumed to be coverage filtered, dense dataframes.
        Calls reindexed to all genomic positions will also work,
        but this function is not the most efficient way to treat such calls.
        Mcalls must have columns: n_meth, n_total
    sample_names
        names, order matched to fps, will be used as columns index
    cpg_gr_df
        must have columns Chromosome, Start, End and consecutive, sorted index
    n_cores
        if > 1, will be parallelized

    Returns
    -------
    dict of dataframes for keys 'is_covered' and 'beta value'
    """

    # avoid alignment issues (just as a precaution)
    if isinstance(sample_names, pd.Series):
        sample_names = sample_names.to_list()
    if isinstance(mcall_df_fps, pd.Series):
        mcall_df_fps = mcall_df_fps.to_list()

    # 0: n_meth, 1: n_total
    # with default backend, there was a UserWarning "memory leak or timeout too short"
    # I haven't investigated it yet, the situation does not arise by setting backend to multiprocessing
    reindexed_sers = list(
        zip(
            *Parallel(n_cores, backend="multiprocessing")(
                delayed(_reindex_meth_calls_to_query_df)(p, cpg_gr_df)
                for p in mcall_df_fps
            )
        )
    )
    res = {
        "n_meth": pd.concat(reindexed_sers[0], keys=sample_names, axis=1),
        "n_total": pd.concat(reindexed_sers[1], keys=sample_names, axis=1),
    }
    res["beta_value"] = res["n_meth"] / res["n_total"]

    return res


# for multiprocessing backend of joblib, this function needs to be top-level
def _reindex_meth_calls_to_query_df(
    mcall_df_p: str, cpg_gr_df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    """Retrieves mcalls for genomic positions in cpg_gr_df, fills missing positions

    Missing positions are filled with np.nan. Currently, this means that columns
    n_total and n_meth are cast to float if there are missing values
    (basically always for sc methylome data).

    The index and row_order of cpg_gr_df is maintained

    Parameters
    ----------
    mcall_df_p
        pickled pd.DataFrame, must have columns Chromosome, Start, End, n_meth, n_total
    cpg_gr_df
        query positions, must have columns Chromosome, Start, End

    Returns
    -------
    (n_meth series, n_total series)

    """

    df = pd.read_pickle(mcall_df_p)
    # pandas merge of categoricals produces object if the categoricals are not identical
    # (pandas 0.25.3)
    # Ensure correct Chromosome dtype in result by setting df dtype to cpg_gr_df dtype
    df["Chromosome"] = df["Chromosome"].astype(cpg_gr_df.Chromosome.dtype)
    # assert that the categorical type cast has not created missing values
    assert df["Chromosome"].notnull().all()

    reindexed_df = pd.merge(cpg_gr_df, df, how="left", on=gr_cols)

    # currently (pandas 0.25.3), the index is discarded during the left merge
    # assert that left merge was performed correctly
    # reset index for both dfs to avoid alignment artifacts
    pd.testing.assert_frame_equal(
        reindexed_df[gr_cols].reset_index(drop=True),
        cpg_gr_df[gr_cols].reset_index(drop=True),
        check_names=False,
    )

    # then we can safely reassign the original index
    reindexed_df.index = cpg_gr_df.index

    return reindexed_df["n_meth"], reindexed_df["n_total"]
