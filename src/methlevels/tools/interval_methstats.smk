"""Retrieve methylation levels for genomic ranges

Args:
    metadata_table: tabular specification of the dataset (methylation calls).
        - must have columns bed_path, name
        - may have additional columns, which will be ignored
        - bed_path is the absolute path to the methylation calling result, which may optionally
          be gzip compressed. it must have the columns n_meth and n_total. Other columns will be ignored
        - name is the sample name which will be used in the output files of the workflow
    region_files: csv list of region files
    names: csv list of names for the region files, *may not have underscores*

Example:
  region_files=cpg-islands=/icgc/dkfzlsdf/analysis/hs_ontogeny/databases/genomic_regions/cpg-islands/regions/no-prefix/cpg-islands_no-prefix.bed,cpg-islands2=/icgc/dkfzlsdf/analysis/hs_ontogeny/databases/genomic_regions/cpg-islands/regions/no-prefix/cpg-islands_no-prefix.bed
  snakemake \
    --snakefile /home/kraemers/projects/methlevels/src/methlevels/tools/interval_methstats.smk \
    --printshellcmd \
    --jobs 24 \
    --forcerun get_meth_stats_by_region_name \
    --config metadata_table=/home/kraemers/temp/metadata.csv \
            region_files="$region_files" \
            output_dir=/icgc/dkfzlsdf/analysis/B080/kraemers/temp/meth_stats \

    --dryrun \

Notes:
- automatically finds the indices of the n_meth and n_total columns. This allows arbitrary BED3+X formats.

to add:
- describe output
"""

# TODO: does not allow underscore in region name, document or improve
# TODO: implement new cli as specified in docstring
# TODO: allow meth calls to be gzipped or not
# TODO: change by_name in by_sample

import gzip
import pandas as pd
from pathlib import Path


# bed or bed.gzip could be used downstream
methlevels_by_name_region = config['output_dir'] + '/{region}/by-name/meth-levels_{name}_{region}.bed'
# feather and bed will also be produced
methlevels_by_region = config['output_dir'] + '/{region}/meth-levels_{region}.p'

# Check and read config variables
metadata_table = pd.read_csv(config['metadata_table'], sep='\t').set_index('name', drop=False)
assert 'bed_path' in metadata_table
assert 'name' in metadata_table
assert Path(metadata_table.iloc[0]['bed_path']).exists()
region_files_dict = dict( s.split('=') for s in config['region_files'].split(',') )

# TODO: this forces gzipped meth calls
# find n_meth and n_unmeth columns
with gzip.open(metadata_table.iloc[0]['bed_path'], 'rt') as fin:
    column_names = next(fin).rstrip().split('\t')
    # bedtools map requires 1-based indices
    n_meth_idx = column_names.index('n_meth') + 1
    n_total_idx = column_names.index('n_total') + 1

wildcard_constraints:
    region = r'[a-zA-Z0-9-]+',
    name = '[a-zA-Z0-9-_]+',

rule all:
    input:
        expand(methlevels_by_name_region, region=region_files_dict.keys(), name=metadata_table['name']),
        expand(methlevels_by_region, region=region_files_dict.keys()),

rule get_meth_stats_by_region_name:
    input:
        calls_bed = lambda wildcards: metadata_table.loc[wildcards.name, 'bed_path'],
        region_bed = lambda wildcards: region_files_dict[wildcards.region],
    params:
        n_meth_idx = n_meth_idx,
        n_total_idx = n_total_idx,
    output:
        methlevels_by_name_region,
    shell:
        """
        echo -e "$(head -n 1 {input.region_bed})\tn_meth\tn_total" > {output}
        bedtools map -a {input.region_bed} -b {input.calls_bed} \
          -c {params.n_meth_idx},{params.n_total_idx} -o sum >> {output}
        """

rule cat_meth_stats_by_region:
    input:
        meth_levels = lambda wildcards: expand(methlevels_by_name_region, name=metadata_table['name'], region=wildcards.region),
    params:
        sample_names = metadata_table['name']
    output:
        pickle = methlevels_by_region,
        feather = methlevels_by_region.replace('.p', '.feather'),
        bed = methlevels_by_region.replace('.p', '.bed'),
    run:
        import pandas as pd

        print('determine index cols')
        df = pd.read_csv(input.meth_levels[0], sep='\t', header=0, nrows=1)
        index_col = list(df.columns)[0:-2]
        print(index_col)

        print('read all individual files')
        def import_df(fp):
            df = pd.read_csv(fp, sep='\t', header=0)
            df = df.set_index(index_col)
            df = df.assign(beta_value = lambda df: df['n_meth'] / df['n_total'])
            df = df[['beta_value', 'n_meth', 'n_total']]
            return df
        dfs = [import_df(fp) for fp in input.meth_levels]
        print(dfs[0].head())
        print(len(dfs))

        print('concat')
        merged_df = pd.concat(dfs, axis=1, keys=params.sample_names)
        print(merged_df.head())

        print('save p')
        merged_df.to_pickle(output.pickle)
        print('save bed')
        merged_df.columns = ['_'.join([sample_name, stat_name]) for sample_name, stat_name in merged_df.columns]
        merged_df.to_csv(output.bed, sep='\t', header=True, index=True)
        merged_df.reset_index().to_feather(output.feather)



# For testing
# =============================
# config = {}
# config['output_dir'] =  '/icgc/dkfzlsdf/analysis/B080/kraemers/temp/meth_stats'
# config['region_files'] = 'cpg-islands=/icgc/dkfzlsdf/analysis/hs_ontogeny/databases/genomic_regions/cpg-islands/regions/no-prefix/cpg-islands_no-prefix.bed,cpg-islands2=/icgc/dkfzlsdf/analysis/hs_ontogeny/databases/genomic_regions/cpg-islands/regions/no-prefix/cpg-islands_no-prefix.bed'

# pids = [
#     'basos_1',
#     'basos_2',
#     'basos_3',
#     'b-cells_1',
#     'b-cells_2',
#     'b-cells_3',
# ]

# pids = [
# 'basos_1',
# 'basos_2',
# 'basos_3',
# 'b-cells_1',
# 'b-cells_2',
# 'b-cells_3',
# 'cdp_1',
# 'cdp_2',
# 'cdp_3',
# 'cdp_4',
# 'cfu-e_1',
# 'cfu-e_2',
# 'cfu-e_3',
# 'clp_1',
# 'clp_2',
# 'clp_3',
# 'cmop_1',
# 'cmop_2',
# 'cmp-cd55-minus_1',
# 'cmp-cd55-minus_2',
# 'cmp-cd55-minus_3',
# 'cmp-cd55-minus_4',
# 'cmp-cd55-plus_1',
# 'cmp-cd55-plus_2',
# 'cmp-cd55-plus_3',
# 'cmp-cd55-plus_4',
# 'cmp-cd55-plus_5',
# 'cmp-cd55-plus_6',
# 'dc-cd11b_1',
# 'dc-cd11b_2',
# 'dc-cd11b_3',
# 'dc-cd8a_1',
# 'dc-cd8a_2',
# 'dc-cd8a_3',
# 'eosinos_1',
# 'eosinos_2',
# 'eosinos_3',
# 'gmp_1',
# 'gmp_2',
# 'gmp_3',
# 'gmp_4',
# 'granulos_1',
# 'granulos_2',
# 'granulos_3',
# 'hsc_1',
# 'hsc_2',
# 'hsc_3',
# 'hsc_4',
# 'hsc_5',
# 'mdp_1',
# 'mdp_2',
# 'megas_1',
# 'megas_2',
# 'megas_3',
# 'mep_1',
# 'mep_2',
# 'mep_3',
# 'mkp_1',
# 'mkp_2',
# 'mkp_3',
# 'mkp_4',
# 'monos_1',
# 'monos_2',
# 'monos_3',
# 'monos-new_1',
# 'monos-new_2',
# 'monos-new_3',
# 'mpp1_1',
# 'mpp1_2',
# 'mpp1_3',
# 'mpp2_1',
# 'mpp2_2',
# 'mpp2_3',
# 'mpp3_1',
# 'mpp3_2',
# 'mpp3_3',
# 'mpp4_1',
# 'mpp4_2',
# 'mpp4_3',
# 'mpp5_1',
# 'mpp5_2',
# 'mpp5_3',
# 'mpp5_4',
# 'mpp5_5',
# 'neutros_1',
# 'neutros_2',
# 'neutros_3',
# 'nk-cells_1',
# 'nk-cells_2',
# 'nk-cells_3',
# 'pdc_1',
# 'pdc_2',
# 'pdc_3',
# 'premege_1',
# 'premege_2',
# 'premege_3',
# 't-cells_1',
# 't-cells_2',
# 't-cells_3',
# 't-cells_4',
# 't-cells_5'
# ]

# metadata_table = pd.DataFrame({
#     'bed_path': expand('/icgc/dkfzlsdf/analysis/B080/kraemers/projects/mbias/sandbox/results_per_pid_july15/{name}/meth/meth_calls/mcalls_{name}_CG_chrom-merged_strands-merged.bed.gz', name=pids),
#     'name': pids})
# metadata_table.head()

# metadata_table.to_csv('/home/kraemers/temp/metadata_hierarchy_meth_calling.csv', sep='\t', header=True, index=False)
