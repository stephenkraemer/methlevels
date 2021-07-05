import pandas as pd
import os
import pyranges as pr
from pandas.api.types import CategoricalDtype
from methlevels.io import gff_to_bed_like_df

"""generate test GTF

!wget http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M10/gencode.vM10.basic.annotation.gtf.gz

gtf_fp = 'gencode.vM10.basic.annotation.gtf.gz'
# get EGF region: chr3:129,675,574-129,757,439
# see IGV screenshot:
gr = pr.read_gtf(gtf_fp)
egf_region_gr = gr['chr3', 129675574:129757439]
target_gtf = os.path.expanduser('~/projects/methlevels/tests/test-data/gencode_mm10_egf_region.gtf')
os.makedirs(os.path.dirname(target_gtf), exist_ok=True)
egf_region_gr.to_gtf(target_gtf)
!head {target_gtf}
"""


egf_region_mm10_gtf = os.path.expanduser('~/projects/methlevels/tests/test-data/gencode_mm10_egf_region.gtf')

def test_read_gencode_df(gtf_fp):

    # TODO: large temporary file
    gtf_fp = egf_region_mm10_gtf
    chrom_dtype = CategoricalDtype(['chr' + str(i) for i in range(19)] + ['chrX', 'chrY', 'chrM'])
    gencode_df = gff_to_bed_like_df(fp = gtf_fp, chrom_dtype = chrom_dtype)
    # gencode_df.to_pickle('/home/stephen/projects/methlevels/tests/test-data/gencode_mm10_egf_region_bed-like-df.p')
    gencode_df_solution = pd.read_pickle('/home/kraemers/projects/methlevels/tests/test-data/gencode_mm10_egf_region_bed-like-df.p')
    pd.testing.assert_frame_equal(gencode_df, gencode_df_solution)
