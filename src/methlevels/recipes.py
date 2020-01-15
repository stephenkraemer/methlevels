"""Useful and exemplary high-level recipes"""

import os
import time
import pickle
import methlevels as ml
import pandas as pd
from typing import Optional, Dict, List


def compute_meth_stats(
    bed_calls: ml.BedCalls,
    intervals: pd.DataFrame,
    result_dir: str,
    filter_args: Optional[Dict] = None,
    root_subject: Optional[str] = None,
    result_name_prefix="",
    result_name_suffix="",
    cores=1,
    additional_index_cols: Optional[List[str]] = None,
) -> Dict[str, Dict[str, str]]:
    """Create complete methylation stats for a set of genomic ranges

    Creates MethStats object and individual dataframes for
    - subject and replicate level
    - element and interval level

    Compute n_meth, n_total, beta_value, zscores.
    If root_subject is given, compute AUC and delta methylation with respect to the
    root subject.

    The output paths for the individual results have the format:
    {results_dir}/{prefix}{hardcoded name}{suffix}
    The prefix and suffix can be used to describe the input data and set version tags.

    Args:
        bed_calls: determines the samples for which meth stats are calculated
        intervals: passed to ml.BedCalls.intersect. See BedCalls doc for details.
        filter_args: Passed too MethStats.filter_intervals.
            Example: dict(coverage=30, n_cpg_min=3, min_delta=0.1)
            Note that without any filtering, there may be undefined beta values,
            which would make filtering in downstream processing code, e.g. for
            clustering, necessary.
        root_subject: if specified, calculate AUC and delta methylation
        additional_index_cols: columns in the intervals df, which should be used as index columns (for MethStats object and dataframes) in addition to the GRanges columns

    Returns:
        Dict detailing the result files. You can also get the dict by calling the _get_output_paths function

    """

    # Possible future improvements:
    # - option to activate defensive check that beta values etc. are not NA
    # - option to determine which stats to calculate (AUC, delta methylation...)

    print("Working on", result_name_prefix)
    print("Writing to", result_dir)

    os.makedirs(result_dir, exist_ok=True)

    output_paths = _get_output_paths(result_dir, result_name_prefix, result_name_suffix)

    assert intervals.columns[0:3].equals(pd.Index(['Chromosome', 'Start', 'End']))
    assert 'region_id' in intervals.columns
    if additional_index_cols is None:
        additional_index_cols = ['region_id']
    elif isinstance(additional_index_cols, list):
        additional_index_cols = ['region_id'] + additional_index_cols
    else:
        raise TypeError()

    print("Calculating Replicate-level MethStats")
    t1 = time.time()
    meth_stats_replevel_v1 = bed_calls.intersect(
        intervals,
        n_cores=cores,
        elements=True,
        additional_index_cols=additional_index_cols,
        drop_additional_index_cols=True,
    )
    print(f"done after {t1 - time.time()}")
    meth_stats_replevel_v1 = meth_stats_replevel_v1.aggregate_element_counts()

    print("Aggregating to Subject-level")
    meth_stats_poplevel_v1 = meth_stats_replevel_v1.convert_to_populations()
    if filter_args is not None:
        print("Filtering intervals")
        meth_stats_poplevel_v1 = meth_stats_poplevel_v1.filter_intervals(**filter_args)
        print("Apply Population-Stats based QC filtering to Rep-level data")
        meth_stats_replevel_v1 = meth_stats_replevel_v1.subset(
            meth_stats_poplevel_v1.counts.index
        )

    # This assertion is mainly intended to check the subsetting of the rep level data
    rep_level_region_ids = meth_stats_replevel_v1.counts.index.get_level_values(
        "region_id"
    )
    pop_level_region_ids = meth_stats_poplevel_v1.counts.index.get_level_values(
        "region_id"
    )
    assert rep_level_region_ids.equals(pop_level_region_ids)

    print("Calculate methylation statistics")
    meth_stats_poplevel_v1.add_beta_value_stat().add_zscores("beta-value")
    if root_subject:
        meth_stats_poplevel_v1.add_deltas(stat_name="beta-value", root=root_subject)
        meth_stats_poplevel_v1.add_auc_stat(root_subject)

    print("Save Rep-level data")
    with open(output_paths["per-replicate"]["meth_stats_obj"], "wb") as fout:
        pickle.dump(meth_stats_replevel_v1, fout)
    meth_stats_replevel_v1.save_flat_elements_df(
        output_paths["per-replicate"]["elements_p"],
        output_paths["per-replicate"]["elements_tsv"],
        output_paths["per-replicate"]["elements_feather"],
    )
    meth_stats_replevel_v1.save_flat_intervals_df(
        output_paths["per-replicate"]["intervals_p"],
        output_paths["per-replicate"]["intervals_tsv"],
        output_paths["per-replicate"]["intervals_feather"],
    )

    print("Save pop-level data")
    with open(output_paths["per-subject"]["meth_stats_obj"], "wb") as fout:
        pickle.dump(meth_stats_poplevel_v1, fout)
    meth_stats_poplevel_v1.save_flat_elements_df(
        output_paths["per-subject"]["elements_p"],
        output_paths["per-subject"]["elements_tsv"],
        output_paths["per-subject"]["elements_feather"],
    )
    meth_stats_poplevel_v1.save_flat_intervals_df(
        output_paths["per-subject"]["intervals_p"],
        output_paths["per-subject"]["intervals_tsv"],
        output_paths["per-subject"]["intervals_feather"],
    )

    return output_paths


def _get_output_paths(
    result_dir: str, prefix: str, suffix: str
) -> Dict[str, Dict[str, str]]:
    """Collect result paths for compute_meth_stats

    Result paths are of the form {result_dir}/{prefix}{hardcoded name}{suffix}.{p,tsv,feather}
    """
    if prefix is not None:
        prefix += "_"
    # fmt: off
    output_paths = {
        "per-replicate": {
            "meth_stats_obj":    result_dir + f"/{prefix}meth-stats-obj_per-replicate{suffix}.p",
            "elements_tsv":      result_dir + f"/{prefix}element_per-replicate{suffix}.tsv",
            "elements_p":        result_dir + f"/{prefix}element_per-replicate{suffix}.p",
            "elements_feather":  result_dir + f"/{prefix}element_per-replicate{suffix}.feather",
            "intervals_tsv":     result_dir + f"/{prefix}interval_per-replicate{suffix}.tsv",
            "intervals_p":       result_dir + f"/{prefix}interval_per-replicate{suffix}.p",
            "intervals_feather": result_dir + f"/{prefix}interval_per-replicate{suffix}.feather",
        },
        "per-subject": {
            "meth_stats_obj":    result_dir + f"/{prefix}meth-stats-obj_per-subject{suffix}.p",
            "elements_tsv":      result_dir + f"/{prefix}element_per-subject{suffix}.tsv",
            "elements_p":        result_dir + f"/{prefix}element_per-subject{suffix}.p",
            "elements_feather":  result_dir + f"/{prefix}element_per-subject{suffix}.feather",
            "intervals_tsv":     result_dir + f"/{prefix}interval_per-subject{suffix}.tsv",
            "intervals_p":       result_dir + f"/{prefix}interval_per-subject{suffix}.p",
            "intervals_feather": result_dir + f"/{prefix}interval_per-subject{suffix}.feather",
        },
    }
    # fmt: on
    return output_paths


# -
