#!/usr/bin/env python3

import numpy as np
import math
np.seterr(all='ignore')

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import contextlib, io, sys

@contextlib.contextmanager
def nostdout():

    '''Prevent print to stdout, but if there was an error then catch it and
    print the output before raising the error.'''

    saved_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    except Exception:
        saved_output = sys.stdout
        sys.stdout = saved_stdout
        print(saved_output.getvalue())
        raise
    sys.stdout = saved_stdout

class T1_fitter(object):

    def __init__(self, ti_vec, t1res=1, t1min=1, t1max=5000, fit_method='mag', ndel=4):
        '''
        ti_vec: vector of inversion times (len(ti_vec) == len(data)
        t1res: resolution of t1 grid-search (in milliseconds)
        t1min,t1max: min/max t1 for grid search (in milliseconds)
        '''
        self.fit_method = fit_method.lower()
        self.t1min = t1min
        self.t1max = t1max
        self.t1res = t1res
        self.ndel = ndel
        if self.fit_method=='nlspr' or self.fit_method=='mag' or self.fit_method=='nls':
            self.init_nls(ti_vec)
        else:
            self.ti_vec = np.array(ti_vec, dtype=np.float)

    def init_nls(self, new_tis=None):
        if new_tis is not None:
            self.ti_vec = np.matrix(new_tis, dtype=np.float)
        #else:
        #    self.ti_vec = np.matrix(self.ti_vec, dtype=np.float)
        n = self.ti_vec.size
        self.t1_vec = np.matrix(np.arange(self.t1min, self.t1max+self.t1res, self.t1res, dtype=np.float))
        self.the_exp = np.exp(-self.ti_vec.T * np.matrix(1/self.t1_vec))
        self.exp_sum = 1. / n * self.the_exp.sum(0).T
        self.rho_norm_vec = np.sum(np.power(self.the_exp,2), 0).T - 1./n*np.power(self.the_exp.sum(0).T,2)

    def __call__(self, d):
        # Work-aropund for pickle's (and thus multiprocessing's) inability to map a class method.
        # See http://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-pythons-multiprocessing-pool-ma
        if self.fit_method=='nlspr':
            return self.t1_fit_nlspr(d)
        elif self.fit_method=='mag':
            return self.t1_fit_magnitude(d)
        elif self.fit_method=='lm':
            return self.t1_fit_lm(d)
        elif self.fit_method=='ctk':
            return self.t1_fit_with_ctk(d)
        elif self.fit_method=='nls':
            return self.t1_fit_nls(d)
        
    def t1_fit_lm(self, data):
        '''
        Finds estimates of T1, a, and b using multi-dimensional
        Levenberg-Marquardt algorithm. The model |c*(1-k*exp(-t/T1))|^2
        is used: only one phase term (c), and data are magnitude-squared.
        The residual is the rms error between the data and the fit.

        INPUT:
        data: the data to estimate from (1d vector)

        RETURNS:
        t1,k,c,residual

        '''
        from scipy.optimize import leastsq
        # Make sure data is a 1d vector
        data = np.array(data.ravel())
        n = data.shape[0]

        # Initialize fit values:
        # T1 tarting value is hard-coded here (TODO: something better! Quick coarse grid search using nlspr?)
        # k should be around 1 - cos(flip_angle) = 2
        # |c| is set to the sqrt of the data at the longest TI
        max_val = (np.abs(data[np.argmax(self.ti_vec)]))
        x0 = np.array([900., 2., max_val])

        predicted = lambda t1,k,c,ti: np.abs( c*(1 - k * np.exp(-ti/t1)) )
        residuals = lambda x,ti,y: y - predicted(x[0], x[1], x[2], ti)
        #err = lambda x,ti,y: np.sum(np.abs(residuals(x,ti,y)))
        x,extra = leastsq(residuals, x0, args=(self.ti_vec.T,data))
        # NOTE: I tried minimize with two different bounded search algorithms (SLSQP and L-BFGS-B), but neither worked very well.
        # An unbounded leastsq fit with subsequent clipping of crazy fit values seems to be the fastest and most robust.
        #x0_bounds = [[0.,5000.],[None,None],[0.,max_val*10.]]
        #res = minimize(err, x0, args=(self.ti_vec.T,data), method='L-BFGS-B', bounds=x0_bounds, options={'disp':False, 'iprint':1, 'maxiter':100, 'ftol':1e-06})

        t1 = x[0].clip(self.t1min, self.t1max)
        k = x[1]
        c = x[2]

        # Compute the residual
        y_hat = predicted(t1, k, c, self.ti_vec)
        residual = np.power(y_hat - data.T, 2).sum()

        # Compute the r-squared
        SS_tot = np.power(data - data.mean(), 2).sum()
        r_squared = 1 - residual/SS_tot
        # r_squared = r_squared.clip(0, 1)

        return(t1,k,c,residual,r_squared)

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
         (c) Board of Trustees, Leland Stanford Junior University.
        See their 2010 MRM paper here: http://www.ncbi.nlm.nih.gov/pubmed/20564597.
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

    def t1_fit_magnitude(self, data):
        if self.ndel > 0 and self.ti_vec.size >= self.ndel + 4:
            indx = data.argmin()      # find the data point closest to the null
            indx_to_del = range(indx - int(np.floor(self.ndel/2)) + 1, indx + int(np.ceil(self.ndel/2)) + 1)
            if indx_to_del[0] >= 0 and indx_to_del[-1] < self.ti_vec.size:
                tis = np.delete(self.ti_vec, indx_to_del)
                data = np.delete(data, indx_to_del)
                for n in range(indx_to_del[0]):
                    data[n] = -data[n]
            else:
                tis = self.ti_vec
                for n in range(indx):
                    data[n] = -data[n]
            fit = T1_fitter(tis, fit_method='mag', t1min=self.t1min, t1max=self.t1max, t1res=self.t1res, ndel=self.ndel)
            (t1, b, a, res) = fit.t1_fit_nls(data)
        else:
            (t1, b, a, res, ind) = self.t1_fit_nlspr(data)
        return (t1, b, a, res)
    
    def t1_fit_with_ctk(self, data):
        '''
        Finds estimates of T1, a, b and slice crosstalk using multi-dimensional
        Trust Region Reflective algorithm. The model |c*(1-k*exp(-t/T1))|
        is used, and corrected for slice crosstalk effect on the magnetization.
        The residual is the rms error between the data and the fit.

        INPUT:
        data: the data to estimate from (1d vector)

        RETURNS:
        t1,k,c,residual,crosstalk

        '''
        from scipy.optimize import least_squares
        # Make sure data is a 1d vector
        data = np.array(data.ravel())
        n = data.shape[0]

        # Initialize fit values:
        # T1 tarting value is hard-coded here (TODO: something better! Quick coarse grid search using nlspr?)
        # k should be around 1 - cos(flip_angle) = 2
        # |c| is set to the sqrt of the data at the longest TI
        max_val = np.max(data)
        min_val = np.min(data)
        x0 = np.array([900., 2., max_val, 0.1])
        lb = [1., 1., min_val, 0.01]
        ub = [5000., 2., max_val*2+1, 0.2]
        
        residuals = lambda x,y: y - (self.t1_model_with_ctk(x[0], x[1], x[2], x[3]))
        lsqresult = least_squares(residuals, x0,  bounds=(lb,ub), loss='linear', args=(data,))

        t1 = lsqresult.x[0].clip(self.t1min, self.t1max)
        k = lsqresult.x[1]
        c = lsqresult.x[2]
        ctk = lsqresult.x[3]

        # Compute the residual
        predicted = self.t1_model_with_ctk(t1, k, c, ctk)
        residual = 1. / np.sqrt(n) * np.sqrt(np.power(1 - predicted / data.T, 2).sum())

        return(t1,k,c,residual,ctk)

    def t1_model_with_ctk(self, t1, k, c, ctk):
        '''
        Compute the corrected magnetization using the T1 model 
        when considering the crosstalk between adjacent slices. 
        The magnetization is calculated using Bloch equation.
        
        '''
        Mz = c*(1 - k * np.exp(-self.ti_vec/t1))
        TR = self.ti_vec[-1] + 2 * (self.ti_vec[1] - self.ti_vec[0]) - self.ti_vec[0]
        half_intleave = int(self.ti_vec.shape[0]/2)
        Mz_corrected = Mz
        Mz_corrected[half_intleave] = Mz[half_intleave] * (1 - 0.5*ctk) + c*(1 - np.exp(-(TR/2)/t1)) * (0.5*ctk)
        Mz_corrected[np.arange(half_intleave+1,self.ti_vec.shape[0])] = Mz[np.arange(half_intleave+1,self.ti_vec.shape[0])] * (1 - ctk) + c * (1 - np.exp(-(TR/2)/t1)) * ctk
        return np.abs(Mz_corrected)

        # Mz = np.abs( c*(1 - k * np.exp(-self.ti_vec/t1)) )
        # half_intleave = int(self.ti_vec.shape[0]/2)
        # crosstalk_correction = np.ones(self.ti_vec.shape[0]).T
        # crosstalk_correction[half_intleave] = 1 - 0.5*ctk
        # crosstalk_correction[np.arange(half_intleave+1,self.ti_vec.shape[0])] = 1 - ctk
        # Mz_corrected = Mz * crosstalk_correction
        # return Mz_corrected
        
def resample(img, pixdim=1.5, ref_file=None):
    d = img.get_data().astype(np.float64)
    # option to align to reference volume
    if ref_file!=None:
        # NOT WORKING! I don't think the dipy registration routine is applying the affine.
        ref = nb.load(ref_file)
        mn = nb.Nifti1Image(d.mean(axis=3), img.affine)
        reg = registration.HistogramRegistration(mn, ref, interp='tri')
        T = reg.optimize('rigid')
        resamp_xform = np.dot(img.affine, T.inv().as_affine())
    else:
        resamp_xform = img.affine
    try:
        from dipy.align.aniso2iso import reslice
    except:
        from dipy.align.aniso2iso import resample as reslice
    data,xform = reslice(d, resamp_xform, img.get_header().get_zooms()[:3], [pixdim]*3, order=5)
    return nb.Nifti1Image(data, xform)

def unshuffle_slices(ni, mux, cal_vols=2, mux_cycle_num=2, ti=None, tr=None, ntis=None, keep=None, descending=False):
    if not ti:
        description = ni._header.get('descrip')
        vals = description.tostring().split(';')
        ti = [int(v[3:]) for v in vals if 'ti' in v][0]
        print("Using TI={:.2f} from description.".format(ti))
    else:
        # ti might be a list, in which case we just need the first ti
        try:
            ti = ti[0]
        except:
            pass
        print("Using TI={:.2f} from argument list.".format(ti))

    if not tr:
        tr = ni._header.get_zooms()[3] * 1000.
    
    if not ntis:
        ntis = int(ni.shape[2] / mux)

    num_cal_trs = mux_cycle_num * mux
    sl_acq = np.zeros((ntis,ntis))
    acq = np.mod(np.arange(ntis-1,-1,-1) - num_cal_trs, ntis)
    for sl in range(ntis):
       sl_acq[sl,:] = np.roll(acq, np.mod(sl,2)*math.ceil(ntis/2.)+int(sl/2)+1)
    
    ti_acq = ti + sl_acq*tr/ntis

    d = ni.get_data()
    d = d[:,:,:,cal_vols:]
    if d.shape[3]<ntis:
        print('WARNING: Too few volumes! zero-padding...')
        sz = list(d.shape)
        zero_pad = ntis - sz[3]
        sz[3] = zero_pad
        d = np.concatenate((d,np.zeros(sz,dtype=float)*np.nan), axis=3)
        #d[...,0:2] = np.zeros((sz[0],sz[1],sz[2],2),dtype=float)*np.nan
    else:
        zero_pad = 0

    tis = np.tile(ti_acq,(mux,int(np.ceil(d.shape[3]/float(ntis)))))
    ntimepoints = d.shape[3]
    d_sort = d[...,ntimepoints-ntis:ntimepoints]
    tis = tis[:,ntimepoints-ntis:ntimepoints]

    # By default assuming descending slice prescription. For ascending slice prescription the TI experienced by slice L is equal to the TI experienced by slice (nslice_per_band - L)
    for sl in range(ntis):
        for m in range(mux):
            slidx = sl + m*ntis
            if descending:
                vidx = np.argsort(tis[sl,:])
            else:
                vidx = np.argsort(tis[ntis-1-sl,:])
            d_sort[:,:,slidx,:] = d_sort[:,:,slidx,vidx]

    ti_sort = np.sort(ti_acq[:,0])
    # The last measurement is junk due to the slice-shuffling
    d_sort = d_sort[...,0:ntis-1]
    ti_sort = ti_sort[0:ntis-1]
    if keep:
        d_sort = d_sort[...,keep]
        ti_sort = ti_sort[keep]
    return d_sort,ti_sort


def main(infile, outbase, mask=None, err_method='lm', fwhm=0.0, t1res=1, t1min=1, t1max=5000, tr=[], ti=[], delete=4, unshuffle=None, keep=[], cal=2, mux_cycle=2, jobs=8, mux=3, pixdim=None, bet_frac=0.5, descending=False):

    import nibabel as nb
    import os
    import sys
    from multiprocessing import Pool
    
    outfiles = {f:outbase+'_'+f+'.nii.gz' for f in ['t1','a','b','res','unshuffled','ctk','r2']}

    ni = nb.load(infile[0])
    if np.array(ti).any():
        tis = ti
        if len(tis) == 1 and tr != None and not unshuffle:
            tis = ti + tr * np.arange(ni.shape[2]/mux - 1) / (ni.shape[2]/mux)
            print("TIs: {}".format(tis.round(1).tolist()))
    elif not unshuffle:
        raise RuntimeError('TIs must be provided on the command line for non-slice-shuffle data!')

    if len(infile) > 1:
        data = np.zeros(ni.shape[0:3]+(len(infile),))
        for i in range(len(infile)):
            ni = nb.load(infile[i])
            data[...,i] = np.squeeze(ni.get_data())
    else:
        if unshuffle:
            data,tis = unshuffle_slices(ni, mux, cal_vols=cal, mux_cycle_num=mux_cycle, ti=ti, tr=tr, keep=keep, descending=descending)
            print('Unshuffled slices, saved to %s ' % outfiles['unshuffled'])
            print('TIs: ', tis.round(1).tolist())
            ni = nb.Nifti1Image(data, ni.affine)
            if pixdim != None:
                print('Resampling data to %0.1fmm^3 ...' % pixdim)
                ni = resample(ni, pixdim)
                data = ni.get_data()
            nb.save(ni, outfiles['unshuffled'])
        else:
            if pixdim != None:
                ni = resample(ni, pixdim)
            data = ni.get_data()

    #data = np.abs(data - 100)
    
    if fwhm>0:
        import scipy.ndimage as ndimage
        sd = np.array(ni._header.get_zooms()[0:3])/fwhm/2.355
        print('Smoothing with %0.1f mm FWHM Gaussian (sigma=[%0.2f,%0.2f,%0.2f] voxels)...' % (tuple([fwhm]+sd.tolist())))
        for i in range(data.shape[3]):
            ndimage.gaussian_filter(data[...,i], sigma=sd, output=data[...,i])

    if mask==None:
        print('Computing mask...')
        mn = np.nanmax(data, axis=3)
        try:
            #from dipy.segment.mask import median_otsu
            #masked_mn, mask = median_otsu(mn, 4, 4)
            from nipype.interfaces import fsl
            fsl.ExtractROI(in_file=infile[0], roi_file=outbase+'_vol.nii.gz', t_min=0,t_size=1).run()
            fsl.BET(in_file=outbase+'_vol.nii.gz', frac=bet_frac, mask=True, out_file=outbase+'_brain').run()
            mask = np.asanyarray(nb.load(outbase+'_brain_mask.nii.gz').dataobj)>0.5
        except:
            print('WARNING: failed to compute a mask. Fitting all voxels.')
            mask = np.ones(mn.shape, dtype=bool)
    elif mask.lower()=='none':
        mask = np.ones((data.shape[0],data.shape[1],data.shape[2]), dtype=bool)
    else:
        mask_ni = nb.load(mask)
        if pixdim != None:
            print('Resampling mask to %0.1fmm^3 ...' % pixdim)
            mask_ni = resample(mask_ni, pixdim)
        mask = np.asanyarray(mask_ni.dataobj)>0.5

    #mask = np.ones_like(data[...,0])  # only when fsl.BET fails
    brain_inds = np.argwhere(mask) # for testing on some voxels: [0:10000,:]
    t1 = np.zeros(mask.shape, dtype=np.float)
    a = np.zeros(mask.shape, dtype=np.float)
    b = np.zeros(mask.shape, dtype=np.float)
    res = np.zeros(mask.shape, dtype=np.float)
    if err_method == 'lm':
        r2  = np.zeros(mask.shape, dtype=np.float)
    if err_method == 'ctk':
        ctk = np.zeros(mask.shape, dtype=np.float)
    
    print('Fitting T1 model...')
    fit = T1_fitter(tis, t1res, t1min, t1max, err_method, delete)

    update_step = 20
    update_interval = round(brain_inds.shape[0]/float(update_step))

    if jobs<2:
        for i,c in enumerate(brain_inds):
            d = data[c[0],c[1],c[2],:]
            nans = np.isnan(d)
            if np.any(nans):
                nn = nans==False
                fit_nan = T1_fitter(tis[nn], t1res, t1min, t1max, err_method, delete)
                if err_method == 'lm':
                    t1[c[0],c[1],c[2]], b[c[0],c[1],c[2]], a[c[0],c[1],c[2]], res[c[0],c[1],c[2]], r2[c[0],c[1],c[2]] = fit_nan(d[nn])
                else:
                    t1[c[0],c[1],c[2]], b[c[0],c[1],c[2]], a[c[0],c[1],c[2]], res[c[0],c[1],c[2]] = fit_nan(d[nn])
            else:
                if err_method == 'lm':
                    t1[c[0],c[1],c[2]], b[c[0],c[1],c[2]], a[c[0],c[1],c[2]], res[c[0],c[1],c[2]], r2[c[0],c[1],c[2]] = fit(d)
                else:
                    t1[c[0],c[1],c[2]], b[c[0],c[1],c[2]], a[c[0],c[1],c[2]], res[c[0],c[1],c[2]] = fit(d)
            if np.mod(i, update_interval)==0:
                progress = int(update_step*i/brain_inds.shape[0]+0.5)
                sys.stdout.write('\r[{0}{1}] {2}%'.format('#'*progress, ' '*(update_step-progress), progress*5))
                sys.stdout.flush()
        print(' finished.')
    else:
        p = Pool(jobs)
        work = [data[c[0],c[1],c[2],:] for c in brain_inds]
        workers = p.map_async(fit, work)
        num_updates = 0
        while not workers.ready():
            i = brain_inds.shape[0] - workers._number_left * workers._chunksize
            if i >= update_interval*num_updates:
                num_updates += 1
                if num_updates<=update_step:
                    sys.stdout.write('\r[{0}{1}] {2}%'.format('#'*num_updates, ' '*(update_step-num_updates), num_updates*5))
                    sys.stdout.flush()

        out = workers.get()
        for i,c in enumerate(brain_inds):
            t1[c[0],c[1],c[2]] = out[i][0]
            b[c[0],c[1],c[2]] = out[i][1]
            a[c[0],c[1],c[2]] = out[i][2]
            res[c[0],c[1],c[2]] = out[i][3]
            if err_method == 'lm':
                r2[c[0],c[1],c[2]]  = out[i][4]
            if err_method == 'ctk':
                ctk[c[0],c[1],c[2]] = out[i][4]

        print(' finished.')

    ni_out = nb.Nifti1Image(t1, ni.affine)
    nb.save(ni_out, outfiles['t1'])
    ni_out = nb.Nifti1Image(a, ni.affine)
    nb.save(ni_out, outfiles['a'])
    ni_out = nb.Nifti1Image(b, ni.affine)
    nb.save(ni_out, outfiles['b'])
    ni_out = nb.Nifti1Image(res, ni.affine)
    nb.save(ni_out, outfiles['res'])
    if err_method == 'lm':
        ni_out = nb.Nifti1Image(r2, ni.affine)
        nb.save(ni_out, outfiles['r2'])
    if err_method == 'ctk':
        ni_out = nb.Nifti1Image(ctk, ni.affine)
        nb.save(ni_out, outfiles['ctk'])

 

if __name__ == '__main__':
   
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.description  = ('Fit T1 using a grid-search.\n\n')
    arg_parser.add_argument('infile', nargs='+', help='path to nifti file with multiple inversion times')
    arg_parser.add_argument('-o', '--outbase', default='./t1fitter', help='path and base filename to output files')
    arg_parser.add_argument('-m', '--mask', help='Mask file (nifti) to use. If not provided, a simple mask will be computed.')
    arg_parser.add_argument('-e', '--err_method', default='lm', help='Error minimization method. Current options are "nlspr"-nonlinear least square with polarity restoration, "mag"-fitting magnitude images without noisy data points, "lm"-Levenberg-Marquardt NLS, "ctk"-a model trying to correct for slice crosstalk effects. Default is lm.')
    arg_parser.add_argument('-f', '--fwhm', type=float, default=0.0, help='FWHM of the smoothing kernel (default=0.0mm = no smoothing)')
    arg_parser.add_argument('-r', '--t1res', type=float, default=1.0, help='T1 grid-search resolution, in ms (default=1.0ms)')
    arg_parser.add_argument('-n', '--t1min', type=float, default=1.0, help='Minimum T1 to allow (default=1.0ms)')
    arg_parser.add_argument('-x', '--t1max', type=float, default=5000.0, help='Maximum T1 to allow (default=5000.0ms)')
    arg_parser.add_argument('--tr', type=float, default=[], help='TR of the slice-shuffled scan (in ms).')
    arg_parser.add_argument('-t', '--ti', type=float, default=[], nargs='+', help='List of inversion times. Must match order and size of input file''s 4th dim. e.g., -t 50.0 400 1200 2400. For slice-shuffed data, you just need to provide the first TI.')
    arg_parser.add_argument('-d', '--delete', type=int, default=4, help='Number of TIs to exclude for fitting T1 (default=4)')
    arg_parser.add_argument('-u', '--unshuffle', action='store_true', help='Unshuffle slices')
    arg_parser.add_argument('-k', '--keep', type=float, default=[], nargs='+', help='indices of the inversion times to use for fitting (default=all)')
    arg_parser.add_argument('-c', '--cal', type=int, default=2, help='Number of calibration volumes for slice-shuffed data (default=2)')
    arg_parser.add_argument('--mux_cycle', type=int, default=2, help='Number of mux calibration cycles (default=2)')
    arg_parser.add_argument('--descending_slices', action='store_true', help='Flag for descending or ascending slices (true=descending, false=ascending')
    arg_parser.add_argument('-j', '--jobs', type=int, default=8, help='Number of processors to run for multiprocessing (default=8)')
    arg_parser.add_argument('-s', '--mux', type=int, default=3, help='Number of SMS bands (mux factor) for slice-shuffeld data (default=3)')
    arg_parser.add_argument('-p', '--pixdim', type=float, default=None, help='Resample to a different voxel size (default is to retain input voxel size)')
    arg_parser.add_argument('-b', '--bet_frac', type=float, default=0.5, help='bet fraction for FSL''s bet function (default is 0.5)')
    args = arg_parser.parse_args()

    main(args.infile, args.outbase, args.mask, args.err_method, args.fwhm, args.t1res, args.t1min, args.t1max, args.tr, args.ti, args.delete, args.unshuffle, args.keep, args.cal, args.mux_cycle, args.jobs, args.mux, args.pixdim, args.bet_frac, args.descending_slices)


