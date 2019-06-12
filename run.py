#!/usr/bin/env python3

import json
import os
import subprocess

def bash_command(cmd):
    subprocess.call(['/bin/bash', '-c', cmd])

if __name__ == '__main__':

    # Parse all of the input arguments from the config.json file
    config_file = '/flywheel/v0/config.json'
    #os.system("more {}".format(config_file))

    # Configure the ENV
    cmd = 'chmod +x /etc/fsl/5.0/fsl.sh;'
    cmd = '{} source /etc/fsl/5.0/fsl.sh;'.format(cmd)
    #cmd = '{} echo ${{FSLDIR}};'.format(cmd)

    if not os.path.isfile(config_file):
        raise AssertionError('No Config File FOUND!')
    else:
        with open(config_file, 'r') as f:
            config = json.load(f)

    infile = config['inputs']['nifti']['location']['path']
    if 'nifti_pe1' in config['inputs']:
        infile_pe1 = config['inputs']['nifti_pe1']['location']['path']
    else:
        infile_pe1 = ''
    TR = config['config']['TR']
    TI = config['config']['TI']
    mux = config['config']['multiband_factor']
    mux_cycle = config['config']['calibration_cycle']
    cal_volume = config['config']['calibration_volume']
    mask_threshold = config['config']['mask_threshold']    
    topup_method = config['config']['topup_method']

    basename = (os.path.basename(infile)).split('.')[0]
    # Set output name
    outdir = '/flywheel/v0/output'
    outpath = os.path.join(outdir, basename)
    if infile_pe1 == '':
        cmd = "{} python3 /flywheel/v0/t1fit_unwarp.py {} {} -b {} --tr {} --ti {} --mux {} --mux_cycle {} --cal {} --method {};".format(cmd, infile, outpath, mask_threshold, TR, TI, mux, mux_cycle, cal_volume, topup_method)
    else:
        cmd = "{} python3 /flywheel/v0/t1fit_unwarp.py {} {} -p {} -b {} --tr {} --ti {} --mux {} --mux_cycle {} --cal {} --method {}".format(cmd, infile, outpath, infile_pe1, mask_threshold, TR, TI, mux, mux_cycle, cal_volume, topup_method)

    #print('{}'.format(cmd))
    bash_command(cmd)
