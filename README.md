# PandoraTargetList

version 0.1.1
July 17, 2023

Repository showing the current ranked list of Pandora targets used by the mission.

``pandora_targets.csv`` contains a list of targets that fall within the parameters for the Pandora mission. Current parameters are:

7.0 < Jmag < 11.5<br>
Hmag < 11.0<br>
Orbital period < 18 days<br>
Host star Teff < 5300 Kelvin<br>
TSM > 1<br>
Assuming 5 scale heights (5H) in calculation of expected spectral feature size<br>
Composite Table of the NASA Exoplanet Archive queried

All column headers correspond to those of the NASA Exoplanet Archive. Further descriptions of these columns can be found at the [documentation pages](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html).

This CSV file was generated using the ``pandora_target`` package developed by Ben Hord. This package can be downloaded and used from [here](https://github.com/benhord/pandora-target)

``pandora_target`` queries either the NASA Exoplanet Archive's Composite or Planetary Systems Tables (the user gets to choose) and downloads all planets from the table. If the Planetary Systems Table is queried, then only the values from the default parameter set is kept rather than a combination of parameter values from different data sets like in the Composite Table. At this point, targets without mass values or those with only mass limits have a mass value calculated according to a modified Chen & Kipping mass-radius relationship. This only differs from [Chen & Kipping, 2016](https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract) for planets with radii larger than 15 Earth radii, where the value is set to 1 Jupiter mass as the mass-radius relation is degenerate in this regime. This is similar to the treatment of mass by the NASA Exoplanet Archive.

Additional parameters that are calculated if not already present are semi-major axis, the ratio of semi-major axis to stellar radius, planetary equilibrium temperature, and the ratio of planet radius to stellar radius. Most planet parameters have an associate ``reflink`` parameter where the provenance of each parameter can be found. Parameters whose values have been calculated will have ``Calculated`` for this parameter.

The transmission spectroscopy values (TSM) of each target are then calculated by using Equation 1 in [Kempton, et al., 2018](https://ui.adsabs.harvard.edu/abs/2018PASP..130k4401K/abstract). Additionally, the size of expected spectral features at H user-defined scale heights is calculated. This is given in ppm. A description of this quantity can also be found in Kempton, et al., 2018.

A final column exists in the CSV file called ``manual_add``. This is a flag that is set to 1 when that particular target has been manually added to the target list rather than programmatically queried from the NASA Exoplanet Archive.
