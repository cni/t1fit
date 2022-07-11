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
    mask_threshold = config['config']['mask_threshold']    
    topup_method = config['config']['topup_method']
    unwarp_direction = config['config']['unwarp_direction']

    if 'nifti_rpe' in config['inputs']:
        infile_pe1 = config['inputs']['nifti_rpe']['location']['path']
    else:
        infile_pe1 = ''
    if 'nifti_B0map_magnitude' in config['inputs']:
        infile_b0map_magnitude = config['inputs']['nifti_B0map_magnitude']['location']['path']
        if 'nifti_B0map_frequency' in config['inputs']:
            infile_b0map_frequency = config['inputs']['nifti_B0map_frequency']['location']['path']
        else:
            raise AssertionError('B0map frequency NIFTI is missing.')
    else:
        infile_b0map_magnitude = ''

    metadata = config['inputs']['nifti']['object']['info']
    try:
        if 'MUXRECON' in metadata:      # mux data reconstructed with muxrecon, 'MUXRECON' contains metadata in the json
            TR  = metadata['MUXRECON']['tr']*1000
            TI  = metadata['MUXRECON']['ti']*1000
            esp = metadata['MUXRECON']['effective_echo_spacing']
            mux = metadata['MUXRECON']['num_bands']
            mux_cycle = metadata['MUXRECON']['num_mux_cal_cycle']
            cal_volume = metadata['MUXRECON']['num_mux_cal_volumes_in_nifti']
        else:     # product hyperband data, metadata parsed by dcm2niix
            TR  = metadata['RepetitionTime']*1000
            TI  = metadata['InversionTime']*1000
            esp = metadata['EffectiveEchoSpacing']
            mux = metadata['MultibandAccelerationFactor']
            mux_cycle = 0
            cal_volume = 0
        # determine descending/ascending slices based on qto matrix and slice timing
        import numpy
        qto = numpy.fromstring(metadata['fslhd']['qto_xyz_matrix'],dtype=float,sep=" ")
        qto_z = qto[8:11]
        qto_zz = qto_z[numpy.argmax(abs(qto_z))]
        slice_time = metadata['SliceTiming'][0]
        print(qto_zz, slice_time)
        if (qto_zz > 0 and slice_time > 0) or (qto_zz < 0 and slice_time == 0):
            descending_slices = False
        elif (qto_zz < 0 and slice_time > 0) or (qto_zz > 0 and slice_time == 0):
            descending_slices = True
    except:  # if no metadata exists then use gear configuration
        TR  = config['config']['TR']
        TI  = config['config']['TI']
        esp = config['config']['esp']
        mux = config['config']['multiband_factor']
        mux_cycle = config['config']['calibration_cycle']
        cal_volume = config['config']['calibration_volume']
        descending_slices = config['config']['descending_slices']
    
    basename = (os.path.basename(infile)).split('.')[0]
    # Set output name
    outdir = '/flywheel/v0/output'
    outpath = os.path.join(outdir, basename)
    if not infile_b0map_magnitude == '':  # use B0 map
        if descending_slices:
            cmd = "{} python3 /flywheel/v0/t1fit_unwarp.py {} {} -b {} --tr {} --ti {} --mux {} --mux_cycle {} --cal {} --esp {} --b0map_flag --b0map_magnitude {} --b0map_frequency {} --unwarpdir {} --descending_slices;".format(cmd, infile, outpath, mask_threshold, TR, TI, mux, mux_cycle, cal_volume, esp, infile_b0map_magnitude, infile_b0map_frequency, unwarp_direction)
        else:
            cmd = "{} python3 /flywheel/v0/t1fit_unwarp.py {} {} -b {} --tr {} --ti {} --mux {} --mux_cycle {} --cal {} --esp {} --b0map_flag --b0map_magnitude {} --b0map_frequency {} --unwarpdir {};".format(cmd, infile, outpath, mask_threshold, TR, TI, mux, mux_cycle, cal_volume, esp, infile_b0map_magnitude, infile_b0map_frequency, unwarp_direction)
    elif not infile_pe1 == '': # use topup
        if descending_slices:
            cmd = "{} python3 /flywheel/v0/t1fit_unwarp.py {} {} -p {} -b {} --tr {} --ti {} --mux {} --mux_cycle {} --cal {} --esp {} --method {} --descending_slices".format(cmd, infile, outpath, infile_pe1, mask_threshold, TR, TI, mux, mux_cycle, cal_volume, esp, topup_method)
        else:
            cmd = "{} python3 /flywheel/v0/t1fit_unwarp.py {} {} -p {} -b {} --tr {} --ti {} --mux {} --mux_cycle {} --cal {} --esp {} --method {}".format(cmd, infile, outpath, infile_pe1, mask_threshold, TR, TI, mux, mux_cycle, cal_volume, esp, topup_method)
    else:  # no fieldmap correction
        if descending_slices:
            cmd = "{} python3 /flywheel/v0/t1fit_unwarp.py {} {} -b {} --tr {} --ti {} --mux {} --mux_cycle {} --cal {} --descending_slices;".format(cmd, infile, outpath, mask_threshold, TR, TI, mux, mux_cycle, cal_volume)
        else:
            cmd = "{} python3 /flywheel/v0/t1fit_unwarp.py {} {} -b {} --tr {} --ti {} --mux {} --mux_cycle {} --cal {};".format(cmd, infile, outpath, mask_threshold, TR, TI, mux, mux_cycle, cal_volume)

    print(cmd)
    bash_command(cmd)
