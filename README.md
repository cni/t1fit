t1fit
=====

T1 fitting algorithms for inversion recovery based measurement, in particular using an IR EPI sequence with variable slice acquisition order based on algorithms described in "Fast T1 mapping using slice-shuffled Simultaneous Multi-Slice inversion recovery EPI" by Hua Wu et al, ISMRM 2015. The sequence was developed for GE 3T scanners and is currently used at the CNI and Lucas Center at Stanford.

E.g., to unshuffle slices and compute T1 estimates from a NIFTI file containing slice-shuffled T1 data:

    ./t1_fitter -u /path/to/nifti.nii.gz

For detailed usage information, run

    ./t1_fitter.py -h

### Correcting for EPI distortion
EPI distortion correction is done using [FSL's topup tool](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup). Provide the NIFTI file of the reversed phase encoding acquisition, along with the NIFTI file of the original phase encoding acquisition, and run the script `t1fit_unwarp.py` with:

    ./t1fit_unwarp.py -p /path/to/reverse/pe/nifti.nii.gz /path/to/nifti.nii.gz outbase

Another option for EPI distortion correction is to use FSL's FUGUE. Provide the NIFTI files of the B0 map magnitude image and the field map (in unit of Hz), and run:

    ./t1fit_unwarp.py --b0map_flag --b0map_magnitude /path/to/B0map/magnitude/nifti.nii.gz --b0map_frequency /path/to/B0map/frequency/nifti.nii.gz /path/to/T1/nifti.nii.gz outbase

Other options include:

    --cal           number of calibration volumes at the beginning of the nifti file (default=2)
    --mux           number of SMS bands (mux factor) for slice-shuffled data (default=3)
    --tr            TR of the slice-shuffled scan in ms (default=3000)
    --ti            shortest inversion time of the slice-shuffled scan in ms (default=50)
    --method        interpolation method for FSL's applytopup, 'jac' or 'lsr' (default is 'jac'). 
                    Refer to FSL's applytopup for further information.
    --unwarp_dir    Direction for unwarping when using B0 map for EPI distortion correction (default is 'y-')

For detailed usage information, run

    ./t1fit_unwarp.py -h

