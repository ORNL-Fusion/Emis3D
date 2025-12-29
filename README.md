# Emis3D_Refactor

This program will fit the synthetic radDist signals compared to experimental data. The
radDists and pre-processed SXR/Bolometer signals should be created prior to using this
routine. Each program requires config files to be run and some specific file organization,
but the general structure is this:


INPUTS within /inputs/{tokamakName}/
------------------------------------------------------------------------------------------------
eqdsks/                           :: The equilbrium files for your runs

radDists/{folder name(s) for run} :: Each folder should contain all of the radDists generated
                                     for that injection location. Multiple injection locations
                                     are supported though including multiple folders within the
                                     run config file

runs/{runID}                      :: Contains the config file and outputs that this program generates

sxr/{shot (or whatever)}          :: Contains the pre-processed SXR signal. Folder name should be identified
                                     within the config file for the run. Note: Please see the README.md within
                                     that folder to see how the data should be organized. 
------------------------------------------------------------------------------------------------



TOKAMAK within tokamaks{tokamakName}/
------------------------------------------------------------------------------------------------
{tokamakName}_settings.yaml :: The settings file for the given tokamak. Contains information such as volume,
                               SXR/Bolometer names, SXR/Bolometer config file names, etc.

CAD_stl_files/              :: The CAD files of the tokamak and various SXR/bolometer arrays

sxrInfo                     :: Folder containig all of the SXR/bolometer configuration files

Other useful stuff that is tokamak dependent:
1. Program to grab/store/massage/ SXR/bolometer data
2. Files used to calibrate SXR/bolometer data to W or W/m2/sr
------------------------------------------------------------------------------------------------



The overall method to run this program is to:
------------------------------------------------------------------------------------------------
1. Create the radDists using the SXR/bolometer arrays that are availble for your shot
2. Grab, pre-process, and save the SXR data
3. Run Emis3D to find the best fit radDist for a given time segment.





Citations:
1. B. Stein-Lubrano et al 2024 Nucl. Fusion 64 036020
2. 