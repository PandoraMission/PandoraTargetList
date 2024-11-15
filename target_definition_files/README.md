# Target Definition Files

## Intro

This directory contains target definition files, priority files, and readout schemes. This information is designed to provide all information targets that is required by the Pandora scheduling software.

## Target categories

There are numerous target categories that are included as directories here. The intention is that each of these is independent, with independent prioritization schemes. So targets are intentionally in multiple categories

The categories are
- time-critical
- primary-exoplanet
- monitoring-standard
- auxiliary-exoplanet
- auxiliary-standard
- occultation-standard
- secondary-exoplanet


target with the **-exoplanet** added to the name should be scheduled as an exoplanet transit target, using the scheduling machenery to ensure that a transit is observed. **-standard** means the target can be observed at any time. The **time-critical** category is for observation that must occur at the specificed time.

The ordered listed here is the approximate order that the scheduler should select these targets.

Below is information on these lists

### time-critical

Obsevations that must occur at a specified time, these should be considered top priority and scheduled first.

### primary-exoplanet

These are the primary science targets, they should be scheduled next. They are directly related to mission success

### monitoring-standard

These targets should be scheduled at a fixed cadence and used for monitoring spacecraft health.

### auxiliary-exoplanet

These targets are transiting exoplanet targets and should be scheduled the same as the primary primary-exoplanet category but should only be used to fill gaps in the schedule

### auxiliary-standard

Targets that should be used to fill gaps in the schedule, they can be scheduled at any time.

### occultation-standard

This list, and only this list, should be used for scheduling targets that fill in gaps when the prime target is occulted by earth.

### secondary-exoplanet

These targets are not used, it is a place to store target definition files for backup primary-exoplanet targets for quickly switching in.

## Readout schemes

There are two files that specify the readout schemes for the two channels
- nirda_readout_schemes.json
- vda_readout_schemes.json

Within these files translate between the mnemonics listed in the target definition files and the specific parameters required in the science calendar.

The readout scheme files are json files that are broken up into a few section:

### metadata

This section contains information on things like version number, last updated date, etc.


### data

This section contain all readout scheme data. Within this section is
- CommandName: this is the name of the Pandora flight software command that these readout schemes are defined to interact with. This parameter should also be used in the science calendar. Potential values are AcquireInfCamImages or AcquireVisCamScienceData.
- IncludedMnemonics: these are the mnemonics names that are defined within the file. 
- FixedParameters: these are parameters that should be used with every readout scheme. They don't depend on readout scheme.
- Mnemonics name: for each value in the list of IncludedMnemonics there is an entry.

these data are used to within the scheduler to generate the payload parameters sections.

## Priority files

These files any prioritization and scheduling information required by the scheduler to add targets to a schedule. So for exoplanet targets, they list the number of remaining transits required.