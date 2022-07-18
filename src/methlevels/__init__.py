import numpy as np


class gr_names:
    start = "Start"
    end = "End"
    chrom = "Chromosome"
    all = ["Chromosome", "Start", "End"]


def assert_gr_index_contract(index):
    assert index.is_lexsorted()
    assert index.names[0:3] == gr_names.all
    assert index.get_level_values("Chromosome").dtype.name == "category"
    assert isinstance(index.get_level_values("Chromosome").categories[0], str)
    assert (
        index.get_level_values("Start").dtype
        == index.get_level_values("End").dtype
        == np.int
    )
    return True


from .methstats import MethStats
from .bed_calls import BedCalls
from .dmr_intervals import DMRIntervals
import methlevels.recipes
from methlevels.plot_region import region_plot
from methlevels.plot_genomic import (
    plot_gene_model,
    plot_genomic_region_track,
    plot_gene_model_get_height,
    plot_genomic_region_track_get_height,
    get_max_text_width_in_x,
    get_max_text_height_width_in_x,
    get_single_row_genomic_track_height,
    add_interval_patches_across_axes,
)
from methlevels.utils import get_max_text_height_in_x_approx_em_square
from methlevels.plot_methlevels import bar_plot
