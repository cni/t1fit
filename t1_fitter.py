#!/usr/bin/env python

import numpy as np

class T1_fitter(object):

    def __init__(self, ti_vec, t1res=1, t1min=1, t1max=5000):
        '''
        ti_vec: vector of inversion times (len(ti_vec) == len(data)
        t1res: resolution of t1 grid-search (in milliseconds)
        t1min,t1max: min/max t1 for grid search (in milliseconds)
        '''
        n = len(ti_vec)
        self.ti_vec = np.matrix(ti_vec, dtype=np.float)
        self.t1_vec = np.matrix(np.arange(t1min, t1max+t1res, t1res, dtype=np.float))
        self.the_exp = np.exp(-self.ti_vec.T * np.matrix(1/self.t1_vec))
        self.exp_sum = 1. / n * self.the_exp.sum(0).T
        self.rho_norm_vec = np.sum(np.power(self.the_exp,2), 0).T - 1./n*np.power(self.the_exp.sum(0).T,2)


    def t1_fit_nls(self, data):
        '''
        Finds estimates of T1, a, and b using a nonlinear least
        squares approach. The model a+b*exp(-t/T1) is used.
        The residual is the rms error between the data and the fit.

        INPUT:
        data: the data to estimate from (1d vector)

        RETURNS:
        t1,b,a,residual

        Based on matlab code written by J. Barral, M. Etezadi-Amoli, E. Gudmundson, and N. Stikov, 2009
         (c) Board of Trustees, Leland Stanford Junior University
        '''
        # Make sure data is a column vector
        data = np.matrix(data.ravel()).T
        n = data.shape[0]

        y_sum = data.sum()

        rho_ty_vec = (data.T * self.the_exp).T - self.exp_sum * y_sum
        # sum(theExp.^2, 1)' - 1/nlsS.N*(sum(theExp,1)').^2;

        # The maximizing criterion
        # [tmp,ind] = max( abs(rhoTyVec).^2./rhoNormVec );
        ind = np.argmax(np.power(np.abs(rho_ty_vec), 2)/self.rho_norm_vec)

        t1_hat = self.t1_vec[0,ind]
        b_hat = rho_ty_vec[ind,0] / self.rho_norm_vec[ind,0]
        a_hat = 1. / n * (y_sum - b_hat * self.the_exp[:,ind].sum())

        # Compute the residual
        model_val = a_hat + b_hat * np.exp(-self.ti_vec / t1_hat)
        # residual = 1/sqrt(nlsS.N) * norm(1 - modelValue./data);
        residual = 1. / np.sqrt(n) * np.sqrt(np.power(1 - model_val / data.T, 2).sum())

        return(t1_hat,b_hat,a_hat,residual)


    def t1_fit_nlspr(self, data):
        '''
        Finds estimates of T1, a, and b using a nonlinear least
        squares approach. The model +-|aMag + bMag*exp(-t/T1)| is used.
        The residual is the rms error between the data and the fit.

        INPUT:
        data: the data to estimate from (1d vector)

        RETURNS:
        t1,b,a,residual

        Based on matlab code written by J. Barral, M. Etezadi-Amoli, E. Gudmundson, and N. Stikov, 2009
         (c) Board of Trustees, Leland Stanford Junior University
        '''
        data = np.matrix(data.ravel()).T
        n = data.shape[0]

        t1 = np.zeros(n)
        b = np.zeros(n)
        a = np.zeros(n)
        resid = np.zeros(n)
        for i in range(n):
            if i>0:
                data[i-1] = -data[i-1]
            (t1[i],b[i],a[i],resid[i]) = self.t1_fit_nls(data)

        ind = np.argmin(resid);

        return(t1[ind],b[ind],a[ind],resid[ind],ind)


def resample(img, pixdim=1.5, ref_file=None):
    d = img.get_data().astype(np.float64)
    # option to align to reference volume
    if ref_file!=None:
        # NOT WORKING! I don't think the dipy registration routine is applying the affine.
        ref = nb.load(ref_file)
        mn = nb.Nifti1Image(d.mean(axis=3), img.get_affine())
        reg = registration.HistogramRegistration(mn, ref, interp='tri')
        T = reg.optimize('rigid')
        resamp_xform = np.dot(img.get_affine(), T.inv().as_affine())
    else:
        resamp_xform = img.get_affine()
    try:
        from dipy.align.aniso2iso import reslice
    except:
        from dipy.align.aniso2iso import resample as reslice
    data,xform = reslice(d, resamp_xform, img.get_header().get_zooms()[:3], [pixdim]*3, order=5)
    return nb.Nifti1Image(data, xform)

def unshuffle_slices(ni, mux, cal_vols=2, ti=None, keep=None):
    if not ti:
        description = ni._header.get('descrip')
        vals = description.tostring().split(';')
        ti = [int(v[3:]) for v in vals if 'ti' in v][0]
        print 'Using TI=%0.2f from description.' % ti
    else:
        # ti might be a list, in which case we just need the first ti
        try:
            ti = ti[0]
        except:
            pass
        print 'Using TI=%0.2f from argument list.' % ti
    tr = ni._header.get_zooms()[3] * 1000.
    ntis = ni.shape[2] / mux
    num_cal_trs = cal_vols * mux
    acq = np.mod(np.arange(ntis-1,-1,-1) - num_cal_trs, ntis)
    sl_acq = np.zeros((ntis,ntis))
    for sl in range(ntis):
        sl_acq[sl,:] = np.roll(acq, np.mod(sl,2)*int(round(ntis/2.))+sl/2+1)

    ti_acq = ti + sl_acq*tr/ntis

    d = ni.get_data()
    d = d[:,:,:,cal_vols:]
    if d.shape[3]<ntis:
        print 'WARNING: Too few volumes! zero-padding...'
        sz = list(d.shape)
        zero_pad = ntis - sz[3]
        sz[3] = zero_pad
        d = np.concatenate((d,np.zeros(sz,dtype=float)*np.nan), axis=3)
    else:
        zero_pad = 0

    tis = np.tile(ti_acq,(mux,np.ceil(d.shape[3]/float(ntis))))

    ntimepoints = d.shape[3]
    d_sort = d[...,ntimepoints-ntis:ntimepoints]
    tis = tis[:,ntimepoints-ntis:ntimepoints]

    for sl in range(ntis*mux):
        indx = np.argsort(tis[sl,:])
        d_sort[:,:,sl,:] = d_sort[:,:,sl,indx]

    ti_sort = np.sort(ti_acq[:,0])
    # The last measurement is junk due to the slice-shuffling
    d_sort = d_sort[...,0:ntis-1]
    ti_sort = ti_sort[0:ntis-1]
    if keep:
        d_sort = d_sort[...,keep]
        ti_sort = ti_sort[keep]
    return d_sort,ti_sort


if __name__ == '__main__':
    import nibabel as nb
    #from dipy.segment.mask import median_otsu
    import os
    import sys
    import argparse

    # To run the fit loop in main, you'll need ip_utils gist:
    try:
        import ip_utils
    except:
        fname = os.path.join(os.path.curdir, 'ip_utils.py')
        try:
            execfile(fname)
        except:
            print('ip_utils not found. Fetching it from github...')
            import urllib, shutil
            local_filename, headers = urllib.urlretrieve('http://gist.github.com/rfdougherty/5548296/raw/ip_utils.py')
            shutil.move(local_filename, fname)
            execfile(fname)

    arg_parser = argparse.ArgumentParser()
    arg_parser.description  = ('Fit T1 using a grid-search.\n\n')
    arg_parser.add_argument('infile', nargs='+', help='path to nifti file with multiple inversion times')
    arg_parser.add_argument('outbase', default='./t1fitter', help='path and base filename to output files')
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
    outfiles = {f:args.outbase+'_'+f+'.nii.gz' for f in ['t1','a','b','res','unshuffled']}

    if args.ti:
        tis = args.ti
    elif not args.unshuffle:
        raise RuntimeError('TIs must be provided on the command line for non-slice-shuffle data!')

    if len(args.infile) > 1:
        ni = nb.load(args.infile[0])
        data = np.zeros(ni.shape[0:3]+(len(args.infile),))
        for i in xrange(len(args.infile)):
            ni = nb.load(args.infile[i])
            data[...,i] = np.squeeze(ni.get_data())
    else:
        ni = nb.load(args.infile[0])
        if args.unshuffle:
            data,tis = unshuffle_slices(ni, args.mux, cal_vols=args.cal, ti=args.ti, keep=args.keep)
            print 'Unshuffled slices, saved to %s. TIs: ' % outfiles['unshuffled'], tis.round(1).tolist()
            ni = nb.Nifti1Image(data, ni.get_affine())
            if args.pixdim != None:
                print('Resampling data to %0.1fmm^3 ...' % args.pixdim)
                ni = resample(ni, args.pixdim)
                data = ni.get_data()
            nb.save(ni, outfiles['unshuffled'])
        else:
            if args.pixdim != None:
                ni = resample(ni, args.pixdim)
            data = ni.get_data()

    if args.fwhm>0:
        import scipy.ndimage as ndimage
        sd = np.array(ni._header.get_zooms()[0:3])/args.fwhm/2.355
        print('Smoothing with %0.1f mm FWHM Gaussian (sigma=[%0.2f,%0.2f,%0.2f] voxels)...' % (tuple([args.fwhm]+sd.tolist())))
        for i in xrange(data.shape[3]):
            ndimage.gaussian_filter(data[...,i], sigma=sd, output=data[...,i])

    if args.mask==None:
        print('Computing mask...')
        mn = np.nanmax(data, axis=3)
        try:
            from dipy.segment.mask import median_otsu
            masked_mn, mask = median_otsu(mn, 4, 4)
        except:
            #mn[np.logical_not(mask)]=0
            mask = ip_utils.get_mask(mn, np.percentile(mn, 50))
    elif args.mask.lower()=='none':
        mask = np.ones((data.shape[0],data.shape[1],data.shape[2]), dtype=bool)
    else:
        mask_ni = nb.load(args.mask)
        if args.pixdim != None:
            print('Resampling mask to %0.1fmm^3 ...' % args.pixdim)
            mask_ni = resample(mask_ni, args.pixdim)
        mask = mask_ni.get_data()>=0.5

    brain_inds = np.argwhere(mask) # for testing on some voxels: [0:10000,:]
    t1 = np.zeros(mask.shape, dtype=np.float)
    a = np.zeros(mask.shape, dtype=np.float)
    b = np.zeros(mask.shape, dtype=np.float)
    res = np.zeros(mask.shape, dtype=np.float)
    print('Fitting T1 model...')
    fit = T1_fitter(tis, args.t1res, args.t1min, args.t1max)

    update_interval = round(brain_inds.shape[0]/20.0)
    for i,c in enumerate(brain_inds):
        d = data[c[0],c[1],c[2],:]
        nans = np.isnan(d)
        if np.any(nans):
            nn = nans==False
            fit_nan = T1_fitter(tis[nn], args.t1res, args.t1min, args.t1max)
            t1[c[0],c[1],c[2]],b[c[0],c[1],c[2]],a[c[0],c[1],c[2]],res[c[0],c[1],c[2]],inflect = fit_nan.t1_fit_nlspr(d[nn])
        else:
            t1[c[0],c[1],c[2]],b[c[0],c[1],c[2]],a[c[0],c[1],c[2]],res[c[0],c[1],c[2]],inflect = fit.t1_fit_nlspr(d)
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

    ni_out = nb.Nifti1Image(t1, ni.get_affine())
    nb.save(ni_out, outfiles['t1'])
    ni_out = nb.Nifti1Image(a, ni.get_affine())
    nb.save(ni_out, outfiles['a'])
    ni_out = nb.Nifti1Image(b, ni.get_affine())
    nb.save(ni_out, outfiles['b'])
    ni_out = nb.Nifti1Image(res, ni.get_affine())
    nb.save(ni_out, outfiles['res'])

