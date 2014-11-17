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


if __name__ == '__main__':
    import nibabel as nb
    #from dipy.segment.mask import median_otsu
    import os
    import sys
    import argparse

    # To run the fit loop in main, you'll need my ip_utils gist:
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
    arg_parser.add_argument('-m', '--mask', help='Mask file (nifti) to use. If not provided, a simple mask will be computed.')
    arg_parser.add_argument('-f', '--fwhm', type=float, default=0.0, help='FWHM of the smoothing kernel (default=0.0mm = no smoothing)')
    arg_parser.add_argument('-r', '--t1res', type=float, default=1.0, help='T1 grid-search resolution, in ms (default=1.0ms)')
    arg_parser.add_argument('-n', '--t1min', type=float, default=1.0, help='Minimum T1 to allow (default=1.0ms)')
    arg_parser.add_argument('-x', '--t1max', type=float, default=5000.0, help='Maximum T1 to allow (default=5000.0ms)')
    arg_parser.add_argument('-t', '--ti', type=float, default=[], nargs='+', help='List of inversion times. Must match order and size of input file''s 4th dim. e.g., -t 50.0 400 1200 2400')
    args = arg_parser.parse_args()

    if len(args.infile) > 1:
        ni = nb.load(args.infile[0])
        d = np.zeros(ni.shape[0:3]+(len(args.infile),))
        for i in xrange(len(args.infile)):
            ni = nb.load(args.infile[i])
            d[...,i] = np.squeeze(ni.get_data())
    else:
        ni = nb.load(args.infile[0])
        d = ni.get_data()
    p,f = os.path.split(args.infile[0])
    basename,ext = (f[0:f.find('.')], f[f.find('.'):])
    outfiles = {f:os.path.join(p,basename+'_'+f+ext) for f in ['t1','a','b','res']}
    tis = args.ti #[50., 280.8, 511.5, 742.3, 973.1, 1203.8, 1434.6, 1665.4, 1896.2, 2126.9, 2357.7, 2588.5]#+230.77

    if args.fwhm>0:
        import scipy.ndimage as ndimage
        sd = np.array(ni._header.get_zooms()[0:3])/args.fwhm/2.355
        print('Smoothing with %0.1f mm FWHM Gaussian (sigma=[%0.2f,%0.2f,%0.2f] voxels)...' % (tuple([args.fwhm]+sd.tolist())))
        for i in xrange(d.shape[3]):
            ndimage.gaussian_filter(d[...,i], sigma=sd, output=d[...,i])

    if args.mask!=None:
        mask = nb.load(args.mask).get_data()
    else:
        print('Computing mask...')
        mn = d.max(3)
        try:
            from dipy.segment.mask import median_otsu
            masked_mn, mask = median_otsu(mn, 4, 4)
        except:
            #mn[np.logical_not(mask)]=0
            mask = ip_utils.get_mask(mn, np.percentile(mn, 50))

    brain_inds = np.argwhere(mask) # for testing on some voxels: [0:10000,:]
    t1 = np.zeros(mask.shape, dtype=np.float)
    a = np.zeros(mask.shape, dtype=np.float)
    b = np.zeros(mask.shape, dtype=np.float)
    res = np.zeros(mask.shape, dtype=np.float)
    print('Fitting T1 model...')
    fit = T1_fitter(tis, args.t1res, args.t1min, args.t1max)

    update_interval = round(brain_inds.shape[0]/20.0)
    for i,c in enumerate(brain_inds):
        t1[c[0],c[1],c[2]],b[c[0],c[1],c[2]],a[c[0],c[1],c[2]],res[c[0],c[1],c[2]],inflect = fit.t1_fit_nlspr(d[c[0],c[1],c[2]])
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

