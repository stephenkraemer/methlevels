# # Imports

from typing import Dict, Iterable, Union, List, Tuple
from pathlib import Path
import signal
from collections import Counter
from attr import attrs
import multiprocessing

import pandas as pd
from joblib import Parallel, delayed

# # Global vars

gr_cols = ["Chromosome", "Start", "End"]

# # Extract and concat mcalls from per-sample mcall files

# ## Main class


class McallsFromPerSampleFiles:
    """

    To terminate, join, close use self.pool.terminate()|.join()|.close()
    """

    def __init__(
        self,
        mcall_df_fps: List[Union[str, Path]],
        sample_names: List[str],
        cpg_gr_df: pd.DataFrame,
        n_cores: int,
    ):

        self.mcall_df_fps = mcall_df_fps
        self.sample_names = sample_names
        self.cpg_gr_df = cpg_gr_df
        self.n_cores = n_cores
        self._result = None

    def start(self):
        self.pool = multiprocessing.Pool(self.n_cores, initializer=init_worker)
        self.async_res_l = [
            self.pool.apply_async(
                _reindex_meth_calls_to_query_df,
                kwds=dict(
                    mcall_df_p=fp,
                    cpg_gr_df=self.cpg_gr_df,
                ),
            )
            for fp in self.mcall_df_fps
        ]

    def display_status_counts(self):
        counter = Counter()
        is_pend = False
        for job in self.async_res_l:
            if not job.ready():
                if is_pend:
                    counter["pend"] += 1
                else:
                    counter["run"] += 1
            else:
                is_pend = False
                if job.successful():
                    counter["done"] += 1
                else:
                    counter["error"] += 1
        print(counter)

    def get(self, raise_on_error=True, keep_individual_results=False):
        if self._result is None:
            individ_res_l = []
            for async_res in self.async_res_l:
                try:
                    individ_res_l.append(async_res.get())
                except Exception as e:
                    if raise_on_error:
                        raise (e)
                    else:
                        individ_res_l.append(str(e))

            if keep_individual_results:
                self.individual_res_l = individ_res_l

            reindexed_sers = list(zip(*individ_res_l))
            concat_res_d = {
                "n_meth": pd.concat(reindexed_sers[0], keys=self.sample_names, axis=1),
                "n_total": pd.concat(reindexed_sers[1], keys=self.sample_names, axis=1),
            }
            concat_res_d["beta_value"] = (
                concat_res_d["n_meth"] / concat_res_d["n_total"]
            ) * 100
            self._result = concat_res_d

        return self._result


# ## Helpers

# ### Extract calls from individual file

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
        check_names=False,  # type: ignore
    )

    # then we can safely reassign the original index
    reindexed_df.index = cpg_gr_df.index

    return reindexed_df["n_meth"], reindexed_df["n_total"]


# ### Utils


def init_worker():
    # Don't interrupt child processes on SIGINT
    signal.signal(signal.SIGINT, signal.SIG_IGN)
