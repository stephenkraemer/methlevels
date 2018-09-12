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

def assert_gr_index_contract(index):
    assert index.is_lexsorted()
    assert index.names[0:3] == gr_names.all
    assert index.get_level_values('Chromosome').dtype.name == 'category'
    assert isinstance(index.get_level_values('Chromosome').categories[0], str)
    assert (index.get_level_values('Start').dtype
            == index.get_level_values('End').dtype
            == np.int)
    return True

from .methstats import MethStats
from .bed_calls import BedCalls
from .plots import DMRPlot
from .dmr_intervals import DMRIntervals