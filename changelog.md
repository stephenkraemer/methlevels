- barplot
  - BREAKING renamed title arg to plot_title
  - BREAKING remove bp_padding, yticks_major, yticks_minor


# next version

- add module io
  - add io.gtf_to_bed_like_df

- remove dpcontracts dependency
  - was only used in methlevels.dmr_intervals, which is - i think - a deprecated experiment anyway
  - removed the dpcontracts code from the module
  - removed dpcontracts from setup.py, requirements etc.

d7051b5
