################################################################################
#
#    Saves arrays of redshifts and cosmological times corresponding to each
#    dataset for later use.
#
################################################################################

import yt
import ytree
import numpy as np

# Load datasets and arbor
ts = yt.load('wise_data/DD00??/output_00??')
a = ytree.load('wise_tree_0_0_0.dat')

# Determine number of datasets.
N = len(ts)

# Loop through halos in the arbor to find one that has a progenitor line going
# all the way to the first dataset. This will be the root halo whose progenitor
# line will be used as reference for redshifts.
for root_halo in a:
    if len(root_halo['prog']) == N:
           break

# There are slight discrepancies between redshifts stored in the datasets and
# redshifts in the saved ytree arbor so save those as separate arrays.
ytree_redshifts = np.zeros(N)
ds_redshifts = np.zeros(N)
cosmological_times = np.zeros(N)

# Loop through each dataset and halo in the progenitor line to save redshifts
# and cosmological time.
# Reverse order ofs progenitor line to go in increasing time.
for i, ds in enumerate(ts):
    halo = root_halo['prog'][::-1][i]
    ytree_redshifts[i] = halo['redshift']
    ds_redshifts[i] = ds.current_redshift
    cosmological_times[i] = ds.current_time.to('yr')

# Save all arrays to directory for later use.
np.save('stored_arrays/ytree_redshifts', ytree_redshifts)
np.save('stored_arrays/ds_redshifts', ds_redshifts)
np.save('stored_arrays/cosmological_times', cosmological_times)


