# popiii_halo_analysis

Code for analysis of Pop III halos. My ultimate objective here is to differentiate between halos that don't form any stars and those that form Pop III stars. This can be broadly broken down into two major steps: gathering halo data; and analysing the halo data.

### Gathering Halo Data

This requires simulation datasets from which to gather halo information as well as merger tree data. These codes save data into a directory that by default I called "stored_arrays" so that directory must exist. They must be run in the following order:

1. redshift_and_time.py
  * Loads datasets and ytree arbor
  * Saves arrays of dataset redshifts, arbor redshifts (which slightly differ from dataset redshifts), dataset cosmological times. Requires *stored_arrays* directory.

2. add_all_halo_fields.py
  * Loads datasets and ytree arbor
  * Loads arbor redshifts array from stored_arrays directory
  * Requires functions from halo_analysis.py
  * Adds all new fields to the halos
  * Saves updated arbor (default name: "full_arbor")

3. nth_nearest_neighbor.py
  * Loads datasets and updated arbor
  * Loads arbor redshifts array from stored_arrays directory
  * Adds 'neighbor_distance' field for all halos
  * Saves arrays of distances between halos for later use if needed

4. assign_popiii_halos.py
  * Loads datasets and updated arbor
  * Loads arbor redshifts array from stored_arrays directory
  * Identifies all Pop III halos and Pop III progenitor halos
  * Adds 'is_popiii_halo' field and 'is_popiii_progenitor' field for all halos
  
### Analysing Halo Data

I use notebooks for data analysis because of convenience for visualization. These notebooks are all independent so any of them can be run without the others. They do require the data to have been gathered from the code above, specifically the updated arbor and the stored arrays.

* redshift_plots.ipynb

  Produces plots of the median, middle 68%, middle 95%, and max and min values of all fields at each redshift for all halos. Also plots Pop III halos and Pop III progenitor halos separately. I wanted to see how statistical properties of halos evolved over time and to determine if there was a clear distinction between progenitor halos and other halos.
  
* halo_pairplots.ipynb

  Produces pairplots of a chosen set of fields for no-star halos and Pop III progenitor halos. Again, I wanted to see if there was a distinction between these two classes of halos, but to view multiple variables together in case there is a more complicated relation in parameter-space.
  
* gaussian_fit.ipynb

  Fits the halo data to a multivariate Gaussian profile then determines the Mahalanobis distances for each halo. This is a generalization of the standard deviation that quantifies how much of an outlier a halo is from the mean.
