#!/bin/bash

# EPI fieldmap correction using FSL FUGUE with pre-computed fieldmap 
# Input: 
#    $1 - 4D EPI time series, 
#    $2 - fmap magnitude, 
#    $3 - fmap fieldmap in Hz, 
#    $4 - echo spacing in second, 
#    $5 - direction of warping (pe1:y-, pe0:y), 
#    $6 - output base name
#
# Input nifti should have standard orientation. Use dcm2niix or fslreorient2std to create these niftis

rest=$1
fmap_mag=$2
fmap_frq=$3
esp=$4
unwarpdir=$5
epi_outbase=$6
fmap_mag_outbase="$6_fmap_mag"
fmap_frq_outbase="$6_fmap_frq"
xfm_fmap="$6_fmap2epi.txt"

echo "Running B0 map correction... "

# unwarp EPI
## convert fieldmap to rad/s
$FSLDIR/bin/fslmaths $fmap_frq -mul 6.283 ${fmap_frq_outbase}_rad
## create brain mask from fmap magitude image, align/resample mask and fieldmap to the EPI image
$FSLDIR/bin/bet $rest ${epi_outbase}_brain -f 0.2
$FSLDIR/bin/bet $fmap_mag ${fmap_mag_outbase}_brain -m -f 0.6  # fmap_mag has low contrast, so need more aggressive brain mask parameter
$FSLDIR/bin/flirt -in ${fmap_mag_outbase}_brain -ref ${epi_outbase}_brain -dof 6 -cost normmi -omat $xfm_fmap 
$FSLDIR/bin/flirt -in ${fmap_frq_outbase}_rad -ref ${epi_outbase}_brain -applyxfm -init $xfm_fmap -out ${fmap_frq_outbase}_rad_resampled
$FSLDIR/bin/flirt -in ${fmap_mag_outbase}_brain_mask -ref ${epi_outbase}_brain -applyxfm -init $xfm_fmap -out ${fmap_mag_outbase}_brain_mask_resampled
## fieldmap correction
$FSLDIR/bin/fugue -i $rest --unwarpdir=$unwarpdir --dwell=$esp --loadfmap=${fmap_frq_outbase}_rad_resampled --mask=${fmap_mag_outbase}_brain_mask_resampled -u ${epi_outbase}_unwarped
#
echo "Completed B0 map correction. Corrected image is saved in ${epi_outbase}_unwarped.nii.gz  "
