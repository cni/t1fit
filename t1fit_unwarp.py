#!/usr/bin/env python

import numpy as np
import nibabel as nb
from nipype.interfaces import fsl
import os
import sys
from t1_fitter import unshuffle_slices
from t1_fitter import main as t1_fit
import warnings
warnings.filterwarnings("ignore") 
np.seterr(all='ignore')

class UnwarpEpi(object):

    def __init__(self, out_basename, num_vols=2):
        self.cal_file = out_basename+'_cal.nii.gz'
        self.acq_file = out_basename+'_acqparams.txt' 
        self.index_file = out_basename+'_index.txt'
        self.topup_out = out_basename+'_topup' 
        self.topup_out_movpar = out_basename+'_topup_movpar.txt'
        self.topup_out_fieldcoef = out_basename+'_topup_fieldcoef.nii.gz'
        self.movpar = None
        self.fieldcoef = None
        self.b0_unwarped = None
        self.num_vols = num_vols

    def prep_data(self, nifti1, nifti2, pe0=None, pe1=None):
        ''' Load the reconstructed image files and generate the files that TOPUP needs. '''
        ni1 = nb.load(nifti1)
        ni2 = nb.load(nifti2)
        ''' Get some info from the nifti headers '''
        phase_dim1 = 1 - ni1.get_header().get_dim_info()[1] # it looks like the phase_dim in nifti header is flipped between 0 and 1 (check mux_recon.py)
        phase_dim2 = 1 - ni2.get_header().get_dim_info()[1]
        if int([s for s in ni1.get_header().__getitem__('descrip').tostring().split(';') if s.startswith('pe=')][0].split('=')[1][0])==1:
            pe_dir1 = 1
        else:
            pe_dir1 = -1
        if int([s for s in ni2.get_header().__getitem__('descrip').tostring().split(';') if s.startswith('pe=')][0].split('=')[1][0])==1:
            pe_dir2 = 1
        else:
            pe_dir2 = -1
        ecsp1 = float([s for s in ni1.get_header().__getitem__('descrip').tostring().split(';') if s.startswith('ec=')][0].split('=')[1])
        readout_time1 = ecsp1 * ni1.shape[phase_dim1] / 1000. # its saved in ms, but we want secs
        ecsp2 = float([s for s in ni2.get_header().__getitem__('descrip').tostring().split(';') if s.startswith('ec=')][0].split('=')[1])
        readout_time2 = ecsp2 * ni2.shape[phase_dim1] / 1000.

        cal1 = [im for i,im in enumerate(nb.four_to_three(ni1)) if i==(self.num_vols-1)]
        cal2 = [im for i,im in enumerate(nb.four_to_three(ni2)) if i==(self.num_vols-1)]

        cal = nb.concat_images(cal1+cal2)
        # Topup requires an even number of slices
        if cal.shape[2]%2:
            d = cal.get_data()
            d = np.concatenate((d,np.zeros((d.shape[0],d.shape[1],1,d.shape[3]), dtype=d.dtype)),axis=2)
            cal = nb.Nifti1Image(d, cal.get_affine())
        nb.save(cal, self.cal_file)

        # Write acquisition parameters to text file acq_file 
        with open(self.acq_file, 'w') as f:
            for i in xrange(len(cal1)):
                row = ['0','0','0',str(readout_time1),'\n']
                row[phase_dim1] = str(pe_dir1)
                f.write(' '.join(row))
            for i in xrange(len(cal2)):
                row = ['0','0','0',str(readout_time2),'\n']
                row[phase_dim2] = str(pe_dir2)
                f.write(' '.join(row))

        for f in [pe0, pe1]:
            if f!=None:
                ni = nb.load(f)
                if ni.get_header().get_data_shape()[2]%2:
                    im = ni.get_data()
                    im = np.concatenate((im,np.zeros((im.shape[0],im.shape[1],1,im.shape[3]), dtype=im.dtype)),axis=2)
                    ni = nb.Nifti1Image(im, ni.get_affine())
                    nb.save(ni, f)

    def run_topup(self):
        topup = fsl.TOPUP()
        topup.inputs.in_file = self.cal_file
        topup.inputs.encoding_file = self.acq_file
        topup.inputs.out_base = self.topup_out
        # The following doesn't seem to help. I guess topup isn't parallelized.
        #topup.inputs.environ = {'FSLPARALLEL':'condor', 'OMP_NUM_THREADS':'12'}
        res = topup.run()
        self.b0_unwarped = res.outputs.out_corrected
        self.fieldcoef = res.outputs.out_fieldcoef
        self.movpar = res.outputs.out_movpar

    def apply_topup(self, in_file, out_base, index, method):
        applytopup = fsl.ApplyTOPUP()
        applytopup.inputs.in_files = in_file
        applytopup.inputs.encoding_file = self.acq_file
        applytopup.inputs.in_index = index
        applytopup.inputs.method   = method
        applytopup.inputs.in_topup_movpar = self.topup_out_movpar
        applytopup.inputs.in_topup_fieldcoef = self.topup_out_fieldcoef
        applytopup.inputs.out_corrected = out_base+'.nii.gz'
        # applytopup.cmdline
        res = applytopup.run()

if __name__ == '__main__':
   
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.description  = ('Fit mono-exponential T1 relaxation model, and run EPI correction when calibration image is provided.\n\n')
    arg_parser.add_argument('infile', help='path to nifti file with multiple inversion times')
    arg_parser.add_argument('outbase', help='basename of the output files')
    arg_parser.add_argument('-p', '--pe1', default='', help='path to nifti file with reverse phase encoding for EPI distortion correction')
    arg_parser.add_argument('-m', '--mask', help='mask file (nifti) to use. If not provided, a simple mask will be computed.')
    arg_parser.add_argument('-b', '--bet_frac', type=float, default=0.5, help='bet fraction for FSL''s bet function (default is 0.5)')
    arg_parser.add_argument('--cal', type=int, default=2, help='number of calibration volumes at the beginning of the nifti file (default=2)')
    arg_parser.add_argument('--tr', type=float, default=3000.0, help='TR of the slice-shuffled scan (in ms, default=3000.0)')
    arg_parser.add_argument('--ti', type=float, default=50.0, help='for slice-shuffled data, provide the first TI (in ms, default=50.0)')
    arg_parser.add_argument('--mux', type=int, default=3, help='number of SMS bands (mux factor) for slice-shuffeld data (default=3)')
    arg_parser.add_argument('--mux_cycle', type=int, default=2, help='Number of mux calibration cycles (default=2)')
    arg_parser.add_argument('--method', type=str, default='jac', help='method for applytopup interpolation. ''jac'' for Jacobian when only one full SS scan (pe0) is done, or ''lsr'' for least-square resampling when both pe0 and pe1 SS scans are done (default is ''jac'')')

    args = arg_parser.parse_args()

    pe0_raw = args.infile
    pe1_raw = args.pe1
    outbase = args.outbase
    cal_vols = args.cal
    ti = args.ti
    tr = args.tr
    mux = args.mux
    method = args.method

    pe0_unshuffled = outbase+'_pe0_unshuffled' 
    pe1_unshuffled = outbase+'_pe1_unshuffled' 
    unwarped   = outbase+'_unwarped'
    t1fit_base = outbase+'_t1fit'

    # unshuffle volumes
    ni0 = nb.load(pe0_raw)
    data, tis = unshuffle_slices(ni0, mux, cal_vols=cal_vols, ti=ti, tr=tr, mux_cycle_num=args.mux_cycle)
    print 'Unshuffled slices, saved to %s. TIs: ' % pe0_unshuffled, tis.round(1).tolist() 
    ni0 = nb.Nifti1Image(data, ni0.get_affine())
    nb.save(ni0, pe0_unshuffled+'.nii.gz')

    # unwarp and fit T1
    if pe1_raw:
        unwarper = UnwarpEpi(outbase, cal_vols)
        if method == 'lsr':
            # when pe1 is provided, unshuffle pe1 data and then unwarp using both pe0 and pe1
            ni1 = nb.load(pe1_raw)
            data, tis = unshuffle_slices(ni1, mux, cal_vols=cal_vols, ti=ti, tr=tr, mux_cycle_num=args.mux_cycle)
            print 'Unshuffled slices, saved to %s.' % pe1_unshuffled
            ni1 = nb.Nifti1Image(data, ni1.get_affine())
            nb.save(ni1, pe1_unshuffled+'.nii.gz')

            print 'Unwarping the unshuffled images using both pe0 and pe1...'
            unwarper.prep_data(pe0_raw, pe1_raw, pe0_unshuffled+'.nii.gz', pe1_unshuffled+'.nii.gz')
            unwarper.run_topup()
            unwarper.apply_topup([pe0_unshuffled+'.nii.gz', pe1_unshuffled+'.nii.gz'], out_base=unwarped, index=[1,2], method=method)

        if method == 'jac':
            print 'Unwarping the unshuffled images for pe0...'
            unwarper.prep_data(pe0_raw, pe1_raw, pe0_unshuffled+'.nii.gz')
            unwarper.run_topup()
            unwarper.apply_topup(pe0_unshuffled+'.nii.gz', out_base=unwarped, index=[1], method=method)

        print 'Fitting T1 maps...'
        t1_fit(infile=[unwarped+'.nii.gz'], outbase=t1fit_base, ti=tis, mask=args.mask, bet_frac=args.bet_frac) 
 
    else:
        # if only pe0 images exist
        print 'No pe1 images provided, fitting T1 without unwarping...'
        t1_fit(infile=[pe0_unshuffled+'.nii.gz'], outbase=t1fit_base, ti=tis, mask=args.mask, bet_frac=args.bet_frac)


