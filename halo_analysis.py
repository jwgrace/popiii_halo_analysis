import yt
import ytree
import numpy as np

def halo_coords(ds, halo):
    '''
    Takes a yt dataset and a halo from a ytree arbor and returns the position
    and virial radius of the halo tied to the dataset.

    Arguments:
    ds - yt dataset
    halo - ytree halo

    Returns:
    pos - 3D YT array of position of center of halo in unitary.
    rad - YT quantity of virial radius of halo in unitary.
      '''
    pos = ds.arr([halo['position_x'], halo['position_y'],
                 halo['position_z']]).to('unitary')
    # virial_radius in ytree is stored as kpc but should be kpccm.
    rad = ds.quan(halo['virial_radius'].v, 'kpccm').to('unitary')

    return pos, rad

def masses(ds, sph, halo=None, is_half=False):
    '''
    Calculates the total mass, gas mass, dark matter mass, stellar mass, and
    Pop III stellar mass, in units of Msun. Also determines the number of live
    Pop III stars and number of Pop III remnants.

    Includes all the mass within a sphere of the virial radius.

    Arguments:
    ds - yt dataset
    sph - yt sphere object
          Sphere with position and radius of chosen halo
    halo - ytree halo
           If no halo is given, function will just return mass values.
           If a halo is given, function will save mass fields to the arbor but
           will not return mass values.
    is_half - boolean
              If False, saves values to the sphere field
              If True, saves values to the half_sphere field.

    Returns:
    tot_mass - yt quantity (Msun)
    gas_mass - yt quantity (Msun)
    dm_mass - yt quantity (Msun)
    stellar_mass - yt quantity (Msun)
    popiii_mass - yt quantity (Msun)
    popiii_num - int
    popiii_rem - int
    '''
    # Get the gas mass and the total mass
    gas_mass, part_mass = sph.quantities.total_mass().to('Msun')
    tot_mass = gas_mass + part_mass

    # Filter the particle masses by type to get the total mass of dark matter,
    # stars, and Pop III stars
    types = sph['particle_type']
    dm_type = 1
    star_type = 7    # star_type does not include pop iii stars
    popiii_type = 5

    # Make array of masses of all particles.
    part_masses = sph['particle_mass'].to('Msun')
    # Filter particles by type.
    dm_masses = part_masses[types == dm_type]
    star_masses = part_masses[types == star_type]
    popiii_masses = part_masses[types == popiii_type]

    # Calculate total masses of each particle type.
    dm_mass = ds.quan(np.sum(dm_masses), 'Msun')
    popiii_mass = ds.quan(np.sum(popiii_masses), 'Msun')
    stellar_mass = ds.quan(np.sum(star_masses), 'Msun') + popiii_mass

    # Pop III stars and their remnants are distiguished by their mass.
    # Once a Pop III star dies, its mass is reduced by a factor of 1e20,
    # so remants are type 5 particles with small mass.
    popiii_num = np.sum(popiii_masses >= .1)
    popiii_rem = np.sum(popiii_masses < .1)

    if halo != None:
        if is_half == False:
            halo['sphere_mass'] = tot_mass
            halo['sphere_gas_mass'] = gas_mass

            halo['sphere_dark_matter_mass'] = dm_mass
            halo['sphere_stellar_mass'] = stellar_mass
            halo['sphere_popiii_mass'] = popiii_mass

            halo['sphere_popiii_number'] = popiii_num
            halo['sphere_popiii_remnants'] = popiii_rem

        else:
            halo['half_sphere_mass'] = tot_mass
            halo['half_sphere_gas_mass'] = gas_mass

            halo['half_sphere_dark_matter_mass'] = dm_mass
            halo['half_sphere_stellar_mass'] = stellar_mass
            halo['half_sphere_popiii_mass'] = popiii_mass

            halo['half_sphere_popiii_number'] = popiii_num
            halo['half_sphere_popiii_remnants'] = popiii_rem

    else:
        return tot_mass, gas_mass, dm_mass, stellar_mass, popiii_mass, \
               popiii_num, popiii_rem

def metallicities(ds, sph, halo=None, is_half=False):
    '''
    Calculates the unitless mean metallicity (metal mass fraction) of the gas
    and of the stars within the virial radius of a halo.

    Calculates the metal fraction by taking the mass-weighted mean of the
    metallicity of the star particles and of the cells.

    Arguments:
    ds - yt dataset
    sph - yt sphere object
          Sphere with position and radius of chosen halo
    halo - ytree halo
           If no halo is given, function will just return metallicity values.
           If a halo is given, function will save metallicity fields to the
           arbor but will not return metallicity values.
    is_half - boolean
              If False, saves values to the sphere field
              If True, saves values to the half_sphere field.

    Returns:
    Z_gas - yt quantity
    Z_star - yt quantity
    '''

    # Filter the particle masses by type to get the total mass of dark matter,
    # stars, and Pop III stars.
    types = sph['particle_type']
    star_type = 7    # star_type does not include pop iii stars
    popiii_type = 5

    # Arrays of particle masses.
    part_masses = sph['particle_mass']
    star_masses = part_masses[types == star_type]
    popiii_masses = part_masses[types == popiii_type]
    # Arrays of particle metallicities
    part_metal_fractions = sph['metallicity_fraction']
    star_metallicities = part_metal_fractions[types == star_type]
    popiii_metallicities = part_metal_fractions[types == popiii_type]
    # Arrays of cell gas masses and metallicities.
    cell_masses = sph['cell_mass']
    cell_metallicities = sph['metallicity']

    # Calulate mass-weighted gas metallicity.
    Z_gas = ds.quan((np.sum(cell_masses*cell_metallicities) \
            / np.sum(cell_masses)), 'dimensionless')
    # Calculate mass-weighted stellar metallicity.
    Z_star = ds.quan(((np.sum(star_masses*star_metallicities) \
                       + np.sum(popiii_masses*popiii_metallicities)) \
                       / (np.sum(star_masses) + np.sum(popiii_masses))),
                       'dimensionless')

    if halo != None:
        if is_half == False:
            halo['sphere_gas_metal_fraction'] = Z_gas
            halo['sphere_stellar_metal_fraction'] = Z_star

        else:
            halo['half_sphere_gas_metal_fraction'] = Z_gas
            halo['half_sphere_stellar_metal_fraction'] = Z_star

    else:
        return Z_gas, Z_star

def masses_and_metallicities(ds, sph, halo=None, is_half=False):
    '''
    Calculates the total mass, gas mass, dark matter mass, stellar mass, and
    Pop III stellar mass within a sphere of the virial radius. Determines the
    number of live Pop III stars and number of Pop III remnants. Calculates the
    unitless mean metallicity (metal mass fraction) of the gas and of the stars.

    Calculates the metal fraction by taking the mass-weighted mean of the
    metallicity of the star particles and of the cells.

    Arguments:
    ds - yt dataset
    sph - yt sphere object
          Sphere with position and radius of chosen halo
    halo - ytree halo
           If no halo is given, function will just return mass and metallicity
           values.
           If a halo is given, function will save mass and metallicity fields to
           the arbor but will not return any values.
    is_half - boolean
              If False, saves values to the sphere field
              If True, saves values to the half_sphere field.

    Returns:
    tot_mass - yt quantity (Msun)
    gas_mass - yt quantity (Msun)
    dm_mass - yt quantity (Msun)
    stellar_mass - yt quantity (Msun)
    popiii_mass - yt quantity (Msun)
    popiii_num - int
    popiii_rem - int
    Z_gas - yt quantity (dimensionless)
    Z_star - yt quantity (dimensionless)
    '''
    # Get the gas mass and the total mass
    gas_mass, part_mass = sph.quantities.total_mass().to('Msun')
    tot_mass = gas_mass + part_mass

    # Filter the particle masses by type to get the total mass of dark matter,
    # stars, and Pop III stars
    types = sph['particle_type']
    dm_type = 1
    star_type = 7    # star_type does not include pop iii stars
    popiii_type = 5

    # Make array of masses of all particles.
    part_masses = sph['particle_mass'].to('Msun')
    # Filter particles by type.
    dm_masses = part_masses[types == dm_type]
    star_masses = part_masses[types == star_type]
    popiii_masses = part_masses[types == popiii_type]

    # Calculate total masses of each particle type.
    dm_mass = ds.quan(np.sum(dm_masses), 'Msun')
    popiii_mass = ds.quan(np.sum(popiii_masses), 'Msun')
    stellar_mass = ds.quan(np.sum(star_masses), 'Msun') + popiii_mass

    # Pop III stars and their remnants are distiguished by their mass.
    # Once a Pop III star dies, its mass is reduced by a factor of 1e20,
    # so remants are type 5 particles with small mass.
    popiii_num = np.sum(popiii_masses >= .1)
    popiii_rem = np.sum(popiii_masses < .1)

    # Arrays of particle metallicities
    part_metal_fractions = sph['metallicity_fraction']
    star_metallicities = part_metal_fractions[types == star_type]
    popiii_metallicities = part_metal_fractions[types == popiii_type]
    # Arrays of cell gas masses and metallicities.
    cell_masses = sph['cell_mass']
    cell_metallicities = sph['metallicity']

    # Calulate mass-weighted gas metallicity.
    Z_gas = ds.quan((np.sum(cell_masses*cell_metallicities) \
            / np.sum(cell_masses)), 'dimensionless')
    # Calculate mass-weighted stellar metallicity.
    Z_star = ds.quan(((np.sum(star_masses*star_metallicities) \
                       + np.sum(popiii_masses*popiii_metallicities)) \
                       / (np.sum(star_masses) + np.sum(popiii_masses))),
                       'dimensionless')

    if halo != None:
        if is_half == False:
            halo['sphere_mass'] = tot_mass
            halo['sphere_gas_mass'] = gas_mass

            halo['sphere_dark_matter_mass'] = dm_mass
            halo['sphere_stellar_mass'] = stellar_mass
            halo['sphere_popiii_mass'] = popiii_mass

            halo['sphere_popiii_number'] = popiii_num
            halo['sphere_popiii_remnants'] = popiii_rem

            halo['sphere_gas_metal_fraction'] = Z_gas
            halo['sphere_stellar_metal_fraction'] = Z_star

        else:
            halo['half_sphere_mass'] = tot_mass
            halo['half_sphere_gas_mass'] = gas_mass

            halo['half_sphere_dark_matter_mass'] = dm_mass
            halo['half_sphere_stellar_mass'] = stellar_mass
            halo['half_sphere_popiii_mass'] = popiii_mass

            halo['half_sphere_popiii_number'] = popiii_num
            halo['half_sphere_popiii_remnants'] = popiii_rem

            halo['half_sphere_gas_metal_fraction'] = Z_gas
            halo['half_sphere_stellar_metal_fraction'] = Z_star

    else:
        return tot_mass, gas_mass, dm_mass, stellar_mass, popiii_mass, \
               popiii_num, popiii_rem, Z_gas, Z_star

def inflow_outflow(ds, sph, pos, rad, halo=None, is_half=False, dr_frac=.05):
    '''
    Calculates the gas inflow and outflow mass and metallicity.

    Mass flow is calculated as a mass flux through a shell centered at the
    virial radius. Inflow and outflow cells are distinguished by a positive or
    negative radial velocity.

    Calculates the dimensionless inflow and outflow metallicity fraction by
    taking the mass-weighted mean of the metallicity of the inflow and outflow
    cells in the shell.

    Arguments:
    ds - yt dataset
    sph - yt sphere object
    pos - yt array
          The (x, y, z) position of the center of the halo in the dataset.
    rad - yt quantity
          The virial radius of the halo.
    halo - ytree halo
    is_half - boolean
              If False, saves values to the sphere field
              If True, saves values to the half_sphere field.
    dr_frac - float
              The fractional thickness of the shell for calculating the mass
              flux. The shell thickness is dr_frac*rad.
    Returns:
    M_in - yt quantity (Msun/yr)
    M_out - yt quantity (Msun/yr)
    Z_in - yt quantity (dimensionless)
    Z_out - yt quantity (dimensionless)
    '''
    # Set the inner and outer bounds of the shell, centered on the radius
    r_out = (1+dr_frac/2)*rad.to('unitary')
    r_in = (1-dr_frac/2)*rad.to('unitary')

    # Create the shell by subtracting an inner sphere from the outer sphere.
    halo_shell = ds.sphere(pos, r_out) - ds.sphere(pos, r_in)

    # Must set center and bulk velocity values of the shell object to properly
    # calulate radial position and velocity.
    bulk_vel = sph.quantities.bulk_velocity()
    halo_shell.set_field_parameter('center', pos)
    halo_shell.set_field_parameter('bulk_velocity', bulk_vel)

    cell_velocities = halo_shell['radial_velocity'].to('unitary/yr')
    # Inflow and outflow distinguished by positive or negative radial velocity.
    outflow = cell_velocities > 0.
    inflow = cell_velocities < 0.

    cell_masses = halo_shell['cell_mass'].to('Msun')
    cell_metallicities = halo_shell['metallicity'].to('code_metallicity')

    # Mass flux is the mass times the velocity divided by the shell thickness.
    M_in = -1/(dr_frac*rad)*np.sum(cell_masses[inflow]*cell_velocities[inflow])
    M_out = 1/(dr_frac*rad)*np.sum(cell_masses[outflow] \
                                   *cell_velocities[outflow])

    # Mass-weighted mean of the inflowing/outflowing metal fraction.
    Z_in = np.sum(cell_metallicities[inflow]*cell_masses[inflow]) \
                  /np.sum(cell_masses[inflow])
    Z_out = np.sum(cell_metallicities[outflow]*cell_masses[outflow]) \
                   /np.sum(cell_masses[outflow])

    if halo != None:
        if is_half == False:
            halo['sphere_inflow_mass'] = M_in
            halo['sphere_outflow_mass'] = M_out
            halo['sphere_inflow_metal_fraction'] = Z_in
            halo['sphere_outflow_metal_fraction'] = Z_out

        else:
            halo['half_sphere_inflow_mass'] = M_in
            halo['half_sphere_outflow_mass'] = M_out
            halo['half_sphere_inflow_metal_fraction'] = Z_in
            halo['half_sphere_outflow_metal_fraction'] = Z_out

    else:
        return M_in, M_out, Z_in, Z_out

def radiation(ds, sph, halo=None, is_half=False):
    '''
    Calculates the volume-averaged Lyman-Werner and Lyman intensity.

    Requires a J21_LW and J_Lyman field to have been added to the dataset.

    Arguments:
    ds - yt dataset
    sph - yt sphere object
          Sphere with position and radius of chosen halo
    halo - ytree halo
           If no halo is given, function will just return radiation intensity
           values.
           If a halo is given, function will save radiation intensity fields to
           the arbor but will not return any values.
    is_half - boolean
              If False, saves values to the sphere field
              If True, saves values to the half_sphere field.

    Returns:
    j21_avg - yt quantity (dimensionless/(s*Hz))
    jlyman_avg - yt quantity (erg/(cm**2*s*Hz))
    '''
    # Arrays of volumes, J21, and J_Lyman for each cell.
    volumes = sph['cell_volume']
    j21 = sph['J21_LW']
    jlyman = sph['J_Lyman']

    # Volume-averaged J21 and J_Lyman flux.
    j21_avg = ds.quan(np.sum(volumes*j21)/np.sum(volumes), \
                      'dimensionless/(s*Hz)')
    jlyman_avg = ds.quan(np.sum(volumes*jlyman)/np.sum(volumes), \
                         'erg/(cm**2*s*Hz)')

    if halo != None:
        if is_half == False:
            halo['sphere_J21_LW'] = j21_avg
            halo['sphere_J_Lyman'] = jlyman_avg

        else:
            halo['half_sphere_J21_LW'] = j21_avg
            halo['half_sphere_J_Lyman'] = jlyman_avg

    return j21_avg, jlyman_avg

def mass_threshold(halo):
    '''
    Calculates the fraction of the theoretical mass threshold for Pop III star formation to occur.
    
    Requires sphere_mass and sphere_J21_LW fields to have been added to the arbor.
    '''
    return (halo['sphere_mass']/(1.25*10**5 + 8.7*10**5*(4*np.pi*halo['sphere_J21_LW'])**.47)).value

def central_density_and_H2(ds, halo, inner_rad=5.):
    '''
    Calculates the mean density and H2 fraction within a small radius of the halo's center.

    Arguments:
    ds - yt dataset
    halo - ytree halo
    inner_rad - float
                The radius of the central portion of the halo in pc

    Returns:
    central_density - yt quantity (g/cm**3)
    central_H2_fraction - yt quantity (dimensionless)
    '''
    
    # Set inner radius: region within which is considered the central portion of the halo.
    # Must be a yt quantity tied to the dataset.
    inner_rad = ds.quan(inner_rad, 'pc')
    
    # Set the position of the halo.
    pos = ds.arr([halo['position_x'], halo['position_y'],
                 halo['position_z']]).to('unitary')
    
    # Create a sphere with the inner radius at the halo's position.
    sph = ds.sphere(pos, inner_rad)
    
    # Calculate the mean density and H2 fraction of the sphere.
    central_density = sph.quantities.weighted_average_quantity('density', 'cell_volume')
    central_H2_fraction = sph.quantities.weighted_average_quantity('H2_fraction', 'cell_mass')

    return central_density, central_H2_fraction

def specific_growth_rate(halo):
    '''
    Calculates the specific growth rate for the given halo.
    
    Specific growth rate is defined here as the mean mass growth rate since the previous data output relative to the most massive progenitor. This growth rate is normalized by the current mass of the halo: (1/M)dM/dt
    
    Requires sphere_mass and cosmological_time fields to be saved to the arbor.
    
    Arguments:
    halo - ytree halo
    
    Returns:
    growth_rate - yt quantity (1/yr)
    '''
    
    # Check length of halo's progenitor line.
    # It must be greater than 1 to have a well-defined growth rate.
    if len(halo['prog']) > 1:
        # Set cosmological time and mass for halo currently.
        current_time = halo['cosmological_time']
        current_mass = halo['sphere_mass'].v
        # Set cosmological time and mass for halo's most massive progenitor.
        ancestor = halo['prog'][1]
        previous_time = ancestor['cosmological_time']
        ancestor_mass = ancestor['sphere_mass'].v

        # Calculate average specific growth rate (growth rate normalized by current mass).
        growth_rate = (current_mass - ancestor_mass)/(current_mass*(current_time - previous_time))
            
    # If the length of the progenitor line is 1, then it is a new halo and doesn't have a well-defined growth rate.
    # Set these to 0 for easy selection.
    else:
        growth_rate = 0.
        
    return growth_rate

def mass_growth_derivative(halo):
    '''
    Calculates the derivative of the mass growth rate, normalized by the current mass of the halo.
    
    Uses a first-order accurate approximation for the derivative of the mass growth rate relative to the line of most massive progenitors. This quantity is normalized by the current mass of the halo: (1/M)d^2M/dt^2
    
    Arguments:
    halo - ytree halo
    
    Returns:
    growth_rate_derivative - yt quantity (1/yr**2)
    '''
    # Check length of halo's progenitor line.
    # It must be greater than 2 for the growth rate to have a well-defined derivative.
    if len(halo['prog']) > 2:
        # Assign previous two ancestors in line of most massive progenitors.
        ancestor1 = halo['prog'][1]
        ancestor2 = halo['prog'][2]
        # Cosmological time and mass of halo currently
        t0 = halo['cosmological_time']
        m0 = halo['sphere_mass'].v
        # Cosmological time and mass one previous step
        t1 = ancestor1['cosmological_time']
        m1 = ancestor1['sphere_mass'].v
        # Cosmological time and mass two previous steps
        t2 = ancestor2['cosmological_time']
        m2 = ancestor2['sphere_mass'].v

        # Most recent mass growth rate
        mass_growth0 = (m0 - m1)/(t0 - t1)
        # Previous growth rate
        mass_growth1 = (m1 - m2)/(t1 - t2)

        # Calculate rate of change of mass growth rates, normalized by current mass.
        growth_rate_derivative = (mass_growth0 - mass_growth1)/(m0*(t0 - t1))
        
        return growth_rate_derivative
        
    # If the length of the progenitor line is less than 3, then there isn't enough mass history to determine a mass growth rate derivative.
    # Set these to 0 for easy selection.
    else:
        return 0.