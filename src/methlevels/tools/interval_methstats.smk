""" Retrieve meth. levels for intervals from BEDs of meth. calls

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

"""

import pandas as pd
from pathlib import Path


# hardcoded output path patterns
methlevels_by_name_region = config['output_dir'] + '/{region}/by-name/meth-levels_{name}_{region}.p'
methlevels_by_region = config['output_dir'] + '/{region}/meth-levels_{region}.p'

# Check and read config variables
metadata_table = pd.read_csv(config['metadata_table'], sep='\t').set_index('name', drop=False)
assert 'bed_path' in metadata_table
assert 'name' in metadata_table
assert Path(metadata_table.iloc[0]['bed_path']).exists()
region_files_dict = dict( s.split('=') for s in config['region_files'].split(',') )


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
    output:
        methlevels_by_name_region,
    shell:
        """
        module load bedtools/2.24.0
        bedtools map -a {input.region_bed} -b {input.calls_bed} -c 8,9 -o sum > {output}
        """

rule cat_meth_stats_by_region:
    input:
        input = lambda wildcards: expand(methlevels_by_name_region, name=metadata_table['name'], region=wildcards.region),
    params:
        sample_names = metadata_table['name']
    output:
        pickle = methlevels_by_region,
        feather = methlevels_by_region.replace('.p', '.feather'),
    run:
        import pandas as pd

        def import_df(fp):
            input_col_names = ['chrom', 'start', 'end', 'n_meth', 'n_total']
            output_col_order = ['beta_value', 'n_meth', 'n_total']  # other cols are index
            df = pd.read_csv(fp, sep='\t',
                             names=input_col_names,
                             index_col=[0, 1, 2])
            df = df.assign(beta_value = lambda df: df['n_meth'] / df['n_total'])
            df = df[output_col_order]
            return df
        dfs = [import_df(fp) for fp in input]
        print(dfs[0].head())

        print(params.sample_names)
        merged_df = pd.concat(dfs, axis=1, keys=params.sample_names)
        print(merged_df.head())

        merged_df.to_pickle(output.pickle)
        merged_df.columns = ['_'.join([sample_name, stat_name]) for sample_name, stat_name in merged_df.columns]
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

# metadata_table = pd.DataFrame({
#     'bed_path': expand('/icgc/dkfzlsdf/analysis/B080/kraemers/projects/mbias/sandbox/results_per_pid_july15/{name}/meth/meth_calls/mcalls_{name}_CG_chrom-merged_strands-merged.bed.gz', name=pids),
#     'name': pids}).set_index('name', drop=False)
# metadata_table

# metadata_table.to_csv('/home/kraemers/temp/metadata.csv', sep='\t')
