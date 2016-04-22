#!/usr/bin/env python

import numpy as np
import t1_fitter
from t1_fitter import T1_fitter

if __name__ == '__main__':
    import nibabel as nb
    #from dipy.segment.mask import median_otsu
    import os
    import sys
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.description  = ('Fit T1 using a grid-search.\n\n')
    arg_parser.add_argument('infile', nargs='+', help='path to nifti file with multiple inversion times')
    arg_parser.add_argument('-o', '--outbase', default='./t1fitter', help='path and base filename to output files')
    arg_parser.add_argument('-m', '--mask', help='Mask file (nifti) to use. If not provided, a simple mask will be computed.')
    arg_parser.add_argument('-f', '--fwhm', type=float, default=0.0, help='FWHM of the smoothing kernel (default=0.0mm = no smoothing)')
    arg_parser.add_argument('-r', '--t1res', type=float, default=1.0, help='T1 grid-search resolution, in ms (default=1.0ms)')
    arg_parser.add_argument('-n', '--t1min', type=float, default=1.0, help='Minimum T1 to allow (default=1.0ms)')
    arg_parser.add_argument('-x', '--t1max', type=float, default=5000.0, help='Maximum T1 to allow (default=5000.0ms)')
    arg_parser.add_argument('-t', '--ti', type=float, default=[], nargs='+', help='List of inversion times. Must match order and size of input file''s 4th dim. e.g., -t 50.0 400 1200 2400. For slice-shuffed data, you just need to provide the first TI.')
    arg_parser.add_argument('-u', '--unshuffle', action='store_true', help='Unshuffle slices')
    arg_parser.add_argument('-k', '--keep', type=float, default=[], nargs='+', help='indices of the inversion times to use for fitting (default=all)')
    arg_parser.add_argument('-c', '--cal', type=int, default=2, help='Number of calibration volumes for slice-shuffed data (default=2)')
    arg_parser.add_argument('-s', '--mux', type=int, default=3, help='Number of SMS bands (mux factor) for slice-shuffeld data (default=3)')
    arg_parser.add_argument('-p', '--pixdim', type=float, default=None, help='Resample to a different voxel size (default is to retain input voxel size)')
    args = arg_parser.parse_args()

    #p,f = os.path.split(args.infile[0])
    #basename,ext = (f[0:f.find('.')], f[f.find('.'):])
    #outfiles = {f:os.path.join(p,basename+'_'+f+ext) for f in ['t1','a','b','res','unshuffled']}
    outfiles = {f:args.outbase+'_'+f+'.nii.gz' for f in ['t1','a','b','res','unshuffled_p','unshuffled_m']}

    if args.ti:
        tis = args.ti
    elif not args.unshuffle:
        raise RuntimeError('TIs must be provided on the command line for non-slice-shuffle data!')

    nim = nb.load(args.infile[0])
    nip = nb.load(args.infile[1])

    if args.unshuffle:
            data_m,tis = t1_fitter.unshuffle_slices(nim, args.mux, cal_vols=args.cal, ti=args.ti, keep=args.keep)
            nim = nb.Nifti1Image(data_m, nim.get_affine())
            data_p,tis = t1_fitter.unshuffle_slices(nip, args.mux, cal_vols=args.cal, ti=args.ti, keep=args.keep)
            nip = nb.Nifti1Image(data_p, nip.get_affine())
            nb.save(nip, outfiles['unshuffled_p'])
            nb.save(nim, outfiles['unshuffled_m'])
            print('Unshuffled slices, saved to %s and %s. TIs: %s'
                  % (outfiles['unshuffled_m'], outfiles['unshuffled_p'], ','.join([str(v) for v in tis.round(1)])))
            if args.pixdim != None:
                print('WARNING: Resampling data to %0.1fmm^3. This may not work well for complex data!' % args.pixdim)
                # Need to resample real and imaginary separately
                data = data_m * np.exp(1j*data_p)
                nim = t1_fitter.resample(nb.Nifti1Image(np.real(data), nim.get_affine()), args.pixdim)
                nip = t1_fitter.resample(nb.Nifti1Image(np.imag(data), nip.get_affine()), args.pixdim)
                data = nim.get_data() + 1j*nip.get_data()
                data_m = np.abs(data)
                data_p = np.angle(data)

    data = data_m * np.exp(1j*data_p)

    if args.fwhm>0:
        import scipy.ndimage as ndimage
        sd = np.array(ni._header.get_zooms()[0:3])/args.fwhm/2.355
        print('WARNING: Smoothing with %0.1f mm FWHM Gaussian (sigma=[%0.2f,%0.2f,%0.2f] voxels). This may not work well for complex data!' % (tuple([args.fwhm]+sd.tolist())))
        for i in xrange(data.shape[3]):
            real = ndimage.gaussian_filter(np.real(data[...,i]), sigma=sd)
            imag = ndimage.gaussian_filter(np.imag(data[...,i]), sigma=sd)
            data[...,i] = real + 1j*imag

    mask_ni = nb.load(args.mask)
    if args.pixdim != None:
        print('Resampling mask to %0.1fmm^3 ...' % args.pixdim)
        mask_ni = t1_fitter.resample(mask_ni, args.pixdim)
    mask = mask_ni.get_data()>=0.5

    brain_inds = np.argwhere(mask) # for testing on some voxels: [0:10000,:]
    t1 = np.zeros(mask.shape, dtype=np.float)
    a = np.zeros(mask.shape, dtype=np.complex)
    b = np.zeros(mask.shape, dtype=np.complex)
    res = np.zeros(mask.shape, dtype=np.complex)
    print('Fitting T1 model...')
    fit = T1_fitter(tis, args.t1res, args.t1min, args.t1max)

    update_interval = round(brain_inds.shape[0]/20.0)
    for i,c in enumerate(brain_inds):
        d = data[c[0],c[1],c[2],:]
        nans = np.isnan(d)
        if np.any(nans):
            nn = nans==False
            fit_nan = T1_fitter(tis[nn], args.t1res, args.t1min, args.t1max)
            t1[c[0],c[1],c[2]],b[c[0],c[1],c[2]],a[c[0],c[1],c[2]],res[c[0],c[1],c[2]] = fit_nan.t1_fit_nls(d[nn])
        else:
            t1[c[0],c[1],c[2]],b[c[0],c[1],c[2]],a[c[0],c[1],c[2]],res[c[0],c[1],c[2]] = fit.t1_fit_nls(d)
        if np.mod(i, update_interval)==0:
            progress = int(20.0*i/brain_inds.shape[0]+0.5)
            sys.stdout.write('\r[{0}{1}] {2}%'.format('#'*progress, ' '*(20-progress), progress*5))
            sys.stdout.flush()
    print(' finished.')

    #from multiprocessing import Pool
    #p = Pool(12)
    #dl = [d[c[0],c[1],c[2]] for c in brain_inds]
    #res = p.map(t1_fit_nlspr, dl, fit.t1_vec, fit.ti_vec, fit.the_exp, fit.exp_sum, fit.rho_norm_vec)
    #for i,c in enumerate(brain_inds):
    #    t1[c[0],c[1],c[2]] = res[i][0]

    ni_out = nb.Nifti1Image(t1, nim.get_affine())
    nb.save(ni_out, outfiles['t1'])
    ni_out = nb.Nifti1Image(np.real(a), nim.get_affine())
    nb.save(ni_out, outfiles['a'])
    ni_out = nb.Nifti1Image(np.real(b), nim.get_affine())
    nb.save(ni_out, outfiles['b'])
    ni_out = nb.Nifti1Image(res, nim.get_affine())
    nb.save(ni_out, outfiles['res'])

