import yt
import ytree
import numpy as np
import numpy.linalg as la

# Load ytree arbor to update.
a = ytree.load('full_arbor/full_arbor.h5')
# Load full time series of datasets.
ts = yt.load('wise_data/DD00??/output_00??')
# Load array of redshifts for the ytree arbor.
redshifts = np.load('stored_arrays/ytree_redshifts.npy')

# Add analysis field to the arbor.
a.add_analysis_field('neighbor_distance', units='kpc')

# Minimum halo mass has ~250 DM particles.
min_mass = 5e5
# Set neighbor number.
n = 5
# all_distances will contain NxN (where N is the number of halos at a given redshift) arrays for each redshift of distances to all other halos.
all_distances = []
# all_neighbor distances will contain the length N arrays of the nth nearest neighbor distance for each redshift.
all_neighbor_distances = []

# Iterate through each redshift and calculate all halo distances at that redshift.
for i, redshift in enumerate(redshifts):

    # Load dataset and list of halos at this redshift.
    ds = ts[i]
    hlist = a.select_halos("tree['tree', 'redshift'] == {}".format(redshift), fields=['redshift'])

    N = len(hlist)

    # Array of positions of each halo.
    positions = np.zeros((N, 3))
    # Create mask to filter only halos above a mass threshold.
    # We only consider distances to halos that are above the mass threshold.
    mass_mask = np.zeros(N, bool)
    
    # Iterate through each halo at this redshift to store its position.
    for j, halo in enumerate(hlist):
        # Set position of halo j in positions array.
        positions[j] = ds.arr([halo['position_x'], halo['position_y'], halo['position_z']]).to('kpc')
        # Mask if halo is above mass threshold.
        if halo['mass'] > min_mass:
            mass_mask[j] = True        

    # Array of distances between halos.
    # redshift_distances[i, j] is the distance between halo i and halo j at this redshift.
    redshift_distances = np.zeros((N,N))
    # Array to store nth neighbor distances for all halos at this redshift.
    redshift_neighbor_distances = np.zeros(N)

    # Iterate through each halo at this redshift and find distances to all other halos.
    for j, halo in enumerate(hlist):
        # First halo position is that of halo j
        pos1 = positions[j]

        # Iterate through all other halos to find distances to each of them.
        # Start with halo j+1 to save time by avoiding double conting.
        for k in range(j+1, N):
            # Second halo position is that of halo k
            pos2 = positions[k]
            
            distance = la.norm(pos1 - pos2)

            # Distance between halos j and k is the same as distance between halos k and j.
            redshift_distances[j,k] = distance
            redshift_distances[k,j] = distance
            
        # Sort the distances to halos above the minimum mass threshold for halo j.
        sorted_redshift_distances = np.sort(redshift_distances[j][mass_mask])
        # Halo j's nth nearest neighbor is element n in its sorted distance array.
        redshift_neighbor_distances[j] = sorted_redshift_distances[n]
        # Save this distance to the halo's field in the arbor.
        halo['neighbor_distance'] = redshift_neighbor_distances[j]
       
    # Append the distance arrays to the complete distance lists.
    all_distances.append(redshift_distances)
    all_neighbor_distances.append(redshift_neighbor_distances)
    
# Save the complete distance lists as arrays.
np.save('stored_arrays/all_distances', np.array(all_distances))
np.save('stored_arrays/all_neighbor_distances', np.array(all_neighbor_distances))
    
# Save the arbor with its updated field values.
a.save_arbor('full_arbor')
