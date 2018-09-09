import numpy as np

class gr_names:
    start = 'Start'
    end = 'End'
    chrom = 'Chromosome'
    all = ['Chromosome', 'Start', 'End']

gr_dtypes_d = dict(
        Start = np.int64,
        End = np.int64,
        Chromosome = str
)

from .methstats import MethStats
from .bed_calls import BedCalls
from .plots import DMRPlot
from .dmr_intervals import DMRIntervals