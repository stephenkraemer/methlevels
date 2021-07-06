import pandas as pd
from pandas.api.types import CategoricalDtype

# * Read and write common file formats

# The io module provides functions for reading and writing common file formats. Files are read into in-memory DataFrame formats which fit into the methlevels API.

# ** GTF Handling

# The gff_to_bed_like_df function reads GTF file formats into a BED-like pandas DataFrame. This DataFrame is suitable for directly generating a PyRanges object. These objects are used for example when annotation genomic region plots.

def gff_to_bed_like_df(fp: str, chrom_dtype: CategoricalDtype):
    """Read GTF file into BED-like dataframe

    *Note: currently only supports GTF format (ie GFF v2).*

    Extract the following BED-like columns into a pandas DataFrame:


    - Chromosome : categorical (as defined by arg chrom_dtype)
    - Start : Int64, converted to 0-based, left-closed intervals
    - End : Int64, converted to 0-based, left-closed intervals
    - gene_name : categorical (dynamically computed categories)
    - score : object
    - Strand : categorical('+', '-')
    - feature : categorical
    - gene_id : categorical (dynamically computed categories)
    - transcript_id : categorical (dynamically computed categories)
    - appris_principal_score : Int64
    - source : categorical
    - frame : object
    - attribute : object

    Parameters
    ----------
    fp
        currently *must* be GTF format file (not GFF3), may be gzipped
    chrom_dtype
        CategoricalDtype with desired sorting order of chromosomes.
        Any chromosomes not defined in chrom_dtype are dropped. If this occurs, a warning is displayed.


    ToDos
    -----
    - support GFF format in different versions (note: GTF format is identical to GFF version 2)
    - consider switching to pyranges GTF reader (fully or as part of this function)
    - consider using less categoricals, this is not a large df and categoricals are clunky for some operations

    """

    strand_dtype = CategoricalDtype(["+", "-"], ordered=True)

    # approx 10s for gencode
    gencode_df = pd.read_csv(
        fp,
        sep="\t",
        header=None,
        comment="#",
        names=[
            "Chromosome",
            "source",
            "feature",
            "Start",
            "End",
            "score",
            "Strand",
            "frame",
            "attribute",
        ],
        dtype={
            "Chromosome": chrom_dtype,
            "Start": "Int64",
            "End": "Int64",
            "Strand": strand_dtype,
            "feature": "category",
            "source": "category",
        },
    )


    # TODO: subtract start, end

    assert (
        gencode_df["Strand"].isin(["+", "-"]).all()
    ), "distance computations require that the strand is defined for all features"

    gencode_df = _expand_gtf_attributes(gencode_df)
    return gencode_df


def _expand_gtf_attributes(df):
    """Extracts metadata from attributes and saves in new dataframe columns
    The following fields are parsed and provided as new columns:
    - gene_id
    - transcript_id
    - gene_name
    - appris_principal_score (float). Features without a principal score are
      assigned a score of 0
    """
    df["gene_id"] = df["attribute"].str.extract('gene_id "(.*?)";').astype('category')
    df["transcript_id"] = df["attribute"].str.extract('transcript_id "(.*?)";').astype('category')
    df["gene_name"] = df["attribute"].str.extract('gene_name "(.*?)";').astype('category')
    df["appris_principal_score"] = (
        df["attribute"].str.extract('tag "appris_principal_(\d)";').astype(float)
    )
    df["appris_principal_score"] = df["appris_principal_score"].fillna(0).astype('Int64')
    return df
