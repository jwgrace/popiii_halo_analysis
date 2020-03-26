import yt
import ytree
import numpy as np
from numpy import linalg as LA
from yt.data_objects.particle_filters import add_particle_filter
import halo_analysis as ha

# Load ytree arbor to update.
a = ytree.load('full_arbor/full_arbor.h5')
# Load full time series of datasets.
ts = yt.load('wise_data/DD00??/output_00??')
# Load array of redshifts for the ytree arbor.
redshifts = np.load('stored_arrays/ytree_redshifts.npy')
# Load cosmological times for each dataset.
cosmological_times = np.load('stored_arrays/cosmological_times.npy')

# Add analysis fields to the arbor.
a.add_analysis_field('is_popiii_halo', units='dimensionless')
a.add_analysis_field('is_popiii_progenitor', units='dimensionless')

# Create particle filter for Pop III stars.
def popiii(pfilter, data):
    particle_type = data[(pfilter.filtered_type, "particle_type")]
    filter = particle_type == 5
    return filter

yt.add_particle_filter("popiii", function=popiii, filtered_type='all', \
                        requires=["particle_type"])

# Set the minimum mass threshold for virialized halos.
# ~250 dark matter particles.
min_mass = 5e5

# Iterate through each dataset, find all Pop III particles, and assign halos to
# each live Pop III particle.
for i, ds in enumerate(ts):

    # Add the Pop III particle filter.
    ds.add_particle_filter('popiii')

    # Load whole dataset to find all Pop III particles in the box.
    ad = ds.all_data()

    # Record formation times for all Pop III particles in this dataset.
    # This will be to determine whether the Pop III particles are new.
    formation_times = ad['popiii', 'creation_time'].to('yr')
    # Record Pop III particle masses.
    # This will be to determine whether the Pop III stars are live or not.
    particle_masses = ad['popiii', 'particle_mass'].to('Msun')

    # After the Pop III star dies, its mass is reduced significantly, so all
    # live Pop III stars will be above a mass threshold.
    live_popiii = particle_masses > .1

    if i != 0:
        # Newly formed Pop III stars formed after the previous dataset.
        new_popiii = formation_times > cosmological_times[i-1]
    else:
        # In the first dataset all Pop III particles are new.
        new_popiii = np.ones(len(formation_times), dtype=bool)

    # We are concerned only with new and live Pop III particles.
    # Create mask to ignore all halos that are not new or live.
    mask = live_popiii + new_popiii

    live_popiii = live_popiii[mask]
    new_popiii = new_popiii[mask]

    # If there are no new or live Pop III particles in this dataset, then move
    # to next dataset.
    if len(mask) == 0:
        continue

    # Get coordinates of all new and live Pop III particles.
    x = ad['popiii', 'particle_position_x'].to('unitary')[mask]
    y = ad['popiii', 'particle_position_y'].to('unitary')[mask]
    z = ad['popiii', 'particle_position_z'].to('unitary')[mask]

    # Create array of positions for each Pop III particle.
    popiii_positions = ds.arr((x, y, z), 'unitary').T

    redshift = redshifts[i]

    # Select all halos at this redshift above mass threshold.
    hlist = a.select_halos("(tree['tree', 'redshift'] == {}) & \
                            (tree['tree', 'mass'] > {})".format(redshift, \
                            min_mass), fields=['redshift', 'mass'])

    # Find halos whose centers are closest to each Pop III particle
    for j, popiii_position in enumerate(popiii_positions):

        # Set initial nearest_distance to be something high so that some halo
        # will be guaranteed to be closer.
        nearest_distance = ds.quan(1., 'unitary')
        # Set initial nearest_halo to None.
        # If no halo is near enough to a Pop III particle, no halo or progenitor
        # are assigned.
        nearest_halo = None

        # Cycle through each halo to compare distance.
        # Halo must be closest halo and the Pop III particle must be within its
        # virial radius.
        for halo in hlist:
            halo_position, halo_radius = ha.halo_coords(ds, halo)
            distance = ds.quan(LA.norm(halo_position - popiii_position), \
                            'unitary')

            if distance < nearest_distance and distance < halo_radius:
                nearest_distance = distance
                nearest_halo = halo

        # If an appropriate halo was found, assign Pop III halo and progenitor
        if nearest_halo != None:
            # Assign Pop III halo if the particle is live.
            if live_popiii[j] == True:
                nearest_halo['is_popiii_halo'] = 1

            # Assign a Pop III progenitor if the particle is new.
            # If it is the first dataset, no progenitor can be assigned.
            if new_popiii[j] == True and i != 0:
                # Some halos don't have any progenitors and so none can be
                # assigned.
                try:
                    # Assign the most massive progenitor to be a progenitor
                    # halo.
                    progenitor = nearest_halo['prog'][1]
                    progenitor['is_popiii_progenitor'] = 1

                except:
                    continue

a.save_arbor('full_arbor')
