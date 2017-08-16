t1fit
=====

T1 fitting algorithms based on algorithms described in "A robust methodology for in vivo T1 mapping" by Barral, Gudmundson, Stikov, Etezadi-Amoli, Stoica and Nishimura, MRM 2010. There are also routines to unshuffle slices from a slice-shuffled pulse sequence, where each slice within a volume experiences a different inversion time. (E.g., the CNI slice-shuffed SMS-EPI sequence.)

E.g., to unshuffle slices and compute T1 estimates from a NIFTI file containing slice-shuffled T1 data:

    ./t1_fitter -u /path/to/nifti.nii.gz

For detailed usage information, run

    ./t1_fitter.py -h

### Correcting for EPI distortion
EPI distortion correction is done using FSL's topup tool. Provide the NIFTI file corresponding to the reversed phase encoding acquisition, along with the NIFTI file of the original phase encoding, and run the script t1fit_unwarp.py with:

    ./t1fit_unwarp.py -p /path/to/reverse/pe/nifti.nii.gz /path/to/nifti.nii.gz outbase
