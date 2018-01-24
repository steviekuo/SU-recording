#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Auto-pilot SU recording for dorsal horn NS cell
 with Keithley 3390 waveform generator; Xcell3 amplifier; NIDAQ"""


import os
import time
import winsound
import datetime
import random
import itertools
import math
import threading
import numpy as np
import nidaqmx as daq
import nidaqmx.constants as cons
import pyvisa as visa


def exp_recd_para():
    # experiment record of time, recording condition
    # stimulation parameter of motor threshold and modification

    local_time = time.asctime(time.localtime(time.time()))

    try:
        exp_rcd_key = ['rat', 'neuron', 'depth', 'angle', 'electrode', 'local_time']
        exp_rcd_val = input(
            '\nRecording info, separated by space\n'
            '<#rat, #neuron, depth, angle, #electrode>\n'
        ).split()
        exp_rcd_val.append(local_time)
        loc_exp_rcd = dict(zip(exp_rcd_key, exp_rcd_val))

        mt_key = ['bp50_mt', 'bp10k_mt', 'sciatic_mt']
        mt_val = input('Mt for BP50, BP10K, sciatic in V\n').split()
        mt_val[:] = list(map(float, mt_val[:]))
        loc_mt = dict(zip(mt_key, mt_val))

    except ValueError:
        print('must input arg separated by space')

    else:
        print('\nexp_record\n'
              '<{5}>\nrat[{0}]/neuron[{1}] at {2}\u00b5m/{3}\u00b0 with #{4} electrode\n'
              .format(*loc_exp_rcd.values()))
        print('bp50_mt={0}, bp10k_mt={1}, sciatic_mt={2}\n'.format(*loc_mt.values()))

    return loc_exp_rcd, loc_mt


def beep(n):
    # makes n beep sound in 1 second

    beep_freq = 1500  # Hz
    beep_dur = int(990/n)  # ms

    for i in range(0, n):
        winsound.Beep(beep_freq, beep_dur)

    return


def countdown(sec, beep_freq=3):
    # display time countdown on screen in seconds

    while sec > 0:
        if sec == 1:
            print('1 sec ', end='\r')
            beep(beep_freq)
            sec -= 1
            print('0 sec ', end='\r')
        else:
            print(sec, ' sec ', end='\r')
            time.sleep(1)
            sec -= 1

    return


def resources():
    # check available resource in current computer

    daq_resource = []
    sys = daq.system.System()  # assign sys as a System object to access property [devices]

    for dev in sys.devices:
        daq_resource.append(dev)

    rm = visa.ResourceManager()

    print(' DAQ resource\n', daq_resource, '\n', 'VISA resource\n', list(rm.list_resources()))

    return


"""I/O to .txt file of bppc test and single-unit recording
"""


def io_su(num, waveform, scs_freq, scs_ontime, scs_mt, scs_mod,
          sciatic_freq, sciatic_pw, status, sciatic_mt, sciatic_mod):

    # make a directory on desktop under current user with name in today's date+_rcd_SWK
    curr_dir_path = os.path.abspath('../..')
    add_dir_name = str(datetime.datetime.now().date())+'_Rcd_SWK'
    new_dir_path = os.path.join(curr_dir_path, 'Desktop', add_dir_name)

    if os.path.isdir(new_dir_path):
        pass
    else:
        os.mkdir(new_dir_path)

    file_name = '{0}_{1}'.format(*exp_rcd.values()) + '_SU.txt'
    file_path = os.path.join(new_dir_path, file_name)
    with open(file_path, 'a') as f:
        f.write('\n<SCS_para #{0}>\n'
                'Trial, waveform, scs_freq, scs_ontime, scs_mt, scs_mod\n'
                '{0}/{1}/{2}/{3}/{4}/{5}\n'
                .format(num, waveform, scs_freq, scs_ontime, scs_mt, scs_mod))
        f.write('<Sciatic para #{0}>\n'
                'Trial, sciatic_freq, sciatic_pw, status, sciatic_mt, sciatic_mod\n'
                '{0}/{1}/{2}/{3}/{4}/{5}\n'
                .format(num, sciatic_freq, sciatic_pw, status, sciatic_mt, sciatic_mod))

    return


def io_char(cell_type, c_thres=0):

    # make a directory on desktop under current user with name in today's date+_rcd_SWK
    curr_dir_path = os.path.abspath('../..')
    add_dir_name = str(datetime.datetime.now().date()) + '_Rcd_SWK'
    new_dir_path = os.path.join(curr_dir_path, 'Desktop', add_dir_name)

    if os.path.isdir(new_dir_path):
        pass
    else:
        os.mkdir(new_dir_path)

    file_name = '{0}_{1}'.format(*exp_rcd.values()) + '_SU.txt'
    file_path = os.path.join(new_dir_path, file_name)

    # decide the proceeding actions depending on bppc test
    c_type = {2: 'a WDR', 1: 'a NS', 0: 'not a typical NS/WDR'}
    with open(file_path, 'a') as f:
        f.write('\n<{5}> rat[{0}]/neuron[{1}\n'
                'exp_record\n'
                'SU recording at {2}\u00b5m/{3}\u00b0 with #{4} electrode\n'
                .format(*exp_rcd.values()))
        f.write('\nCharacterization: This is {0} neuron\n'.format(c_type[cell_type]))
        f.write('C-fiber threshold={0}X sciatic MT\n'.format(c_thres))
        f.write('SU_record file:\n{0}\n'.format(new_dir_path))

    return


"""Configure Keithley 3390 to be ready to output SCS current, 
   triggered by TTL signal from NI-DAQ 6126 with logic high threshold > 2.5 V 
"""


def sample_trigger():

    with daq.Task() as task:
        task.do_channels.add_do_chan('Dev1/port1/line0')     # PFI 0 /P1.0

        print('\nsampling start...\n')

        task.start()
        task.write([True, False])
        task.stop()

    return


def char():
    # total 270=30*9sec

    char_list = ['4g', '6g', '10g', '15g', '26g', 'BRUSH', 'PRESS', 'PINCH', 'CRUSH']

    for i in char_list:
        print('wait 10sec, prepare for {0}'.format(i))
        countdown(10, 3)
        print('start {0} for 10 secs'.format(i))
        countdown(10, 5)
        print('rest 10 secs')
        countdown(10, 1)

    return


def wavegen(waveform, scs_freq, scs_ontime, scs_mt, scs_mod):
    # Keithley 3390 configuration to be triggered/gated by NI-DAQ 6216 BNC with logic signal

    rm = visa.ResourceManager()  # assign NI backend as resource manager
    keithley = rm.open_resource('usb0::0x05E6::0x3390::1425019::INSTR')  # open keithley
    keithley.timeout = None
    keithley.write('*rst; *cls')  # reset to factory default, clear status command

    # define arbitrary waveform, burst/trigger, frequency/amplitude
    if waveform == 'Sin_10K':
        keithley.write('function sin')
    else:
        keithley.write('function user')
        keithley.write('function:user {0}'.format(str(waveform)))

    keithley.write('burst:state on')

    """trigger mode"""
    keithley.write('burst:mode triggered')
    keithley.write('burst:ncycles {0}'.format('infinity'))   # or str(scs_freq * scs_ontime)

    """ gated mode
    keithley.write('burst:mode gated')
    keithley.write('burst:gate:polarity normal')
    """

    keithley.write('trigger:source external')
    keithley.write('trigger:slope positive')

    keithley.write('frequency {0}'.format(str(scs_freq)))
    keithley.write('voltage:unit vpp')
    keithley.write('voltage {0}'.format(str(scs_mt * scs_mod / 100)))

    keithley.close()  # close the instrument handle session

    return


def sciatic(status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw, sciatic_ontime=60, mp_bp=-1):
    # time unit = 1ms; SI from 100us/V => 1mA/V, bp=-1 mp=1
    """
    def time_pad_gen(status, freq, pw, on_time=60):
        time_pad_unit = [amp * status] * 300 + [0] * 700  # 1s template
        time_pad = time_pad_unit * 60  # 60s time pad
        return time_pad
    """

    switch = {1: 'on', 0: 'off'}  # 0.5:25x, 1.5:75x, 2: 100x  3x c-threshold~30x

    num_si = 2.0

    amp = status * sciatic_mt * sciatic_mod / 10  # 10(mA/V)=[1*1(100uA/V)*100(mod)]/10
    if amp > 10:  # max amp = 10 for A-M2000 5Vpp*2, Caputron 10Vpp*1
        amp = 10
    output_comm = round(amp/num_si, 2)

    step = 100  # us
    pad = (1/sciatic_freq) * (10 ** 6) / step  # length(1/freq) in 100us
    sample_per_second = float((10 ** 6) / step)  # specify sample per second for NI
    scale_wave = np.zeros(int(pad))
    scale_pw = math.floor(sciatic_pw/step)  # covert pw from 300us to 3 points in pad
    scale_wave[:scale_pw] = output_comm
    scale_wave[scale_pw:2*scale_pw] = mp_bp * output_comm

    with daq.Task() as task:
        task.ao_channels.add_ao_voltage_chan('Dev1/ao0', min_val=-10.0, max_val=10.0
                                             , units=cons.VoltageUnits.VOLTS)
        task.timing.cfg_samp_clk_timing(sample_per_second, sample_mode=cons.AcquisitionType.FINITE
                                        , samps_per_chan=len(scale_wave))
        task.out_stream.output_buf_size = len(scale_wave)+1
        '''
        print(task.channels.physical_channel)
        print(daq.system.Device('Dev1').terminals)

        task.triggers.start_trigger.trig_type = cons.TriggerType.DIGITAL_EDGE
        task.triggers.start_trigger.cfg_dig_edge_start_trig('/Dev1/PFI0', trigger_edge=cons.Edge.RISING)
        task.out_stream.relative_to = cons.WriteRelativeTo.FIRST_SAMPLE
        task.out_stream.offset = 0

        task.triggers.pause_trigger.trig_type = cons.TriggerType.DIGITAL_LEVEL
        task.triggers.pause_trigger.dig_lvl_src = '/Dev1/PFI1'
        task.triggers.pause_trigger.dig_lvl_when = cons.Level.HIGH
        '''

        num_sci_stim = round(sciatic_ontime * sciatic_freq)  # i=num of stim within sciatic_ontime
        print('Sciatic stimulation is {0} (amp = {1}mA) for {2}sec\n'
              .format(switch[status], amp, sciatic_ontime))
        task.write(scale_wave)

        beep(3)

        while num_sci_stim > 0:
            task.start()
            task.wait_until_done(10)  # wait all analog output read/write
            task.stop()
            num_sci_stim -= 1

        beep(1)

    return


def scs(waveform, scs_freq, scs_mod, scs_mt, scs_ontime):

    rm = visa.ResourceManager()  # assign NI backend as resource manager
    keithley = rm.open_resource('usb0::0x05E6::0x3390::1425019::INSTR')  # open keithley
    keithley.write('OUTput ON')

    with daq.Task() as task:
        task.do_channels.add_do_chan('Dev1/port1/line0')  # PFI 0 /P1.0

        beep(5)
        print('\n{0} stimulation at {1}Hz, {2}% MT: {3}\n'
              .format(waveform, scs_freq, scs_mod, scs_mt))
        # task.start()
        task.write([True, False], auto_start=True)
        countdown(scs_ontime, 1)
        keithley.write('OUTput OFF')
        task.stop()

    return


def su(num, waveform, scs_freq, scs_ontime, scs_mt, scs_mod,
       status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw):

    io_su(num, waveform, scs_freq, scs_ontime, scs_mt, scs_mod,
          status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw)

    # check inputs/raise error before exec
    # trigger sampling at t=0 for 60 sec
    t = threading.Timer(0, sample_trigger)
    t.start()
    t.join()
    # start episode at t=0
    threading.Timer(0, wavegen, [waveform, scs_freq, scs_ontime, scs_mt, scs_mod]).start()
    threading.Timer(0, sciatic, [status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw]).start()
    # start scs stim at t=15, for 30s
    threading.Timer(15, scs, [waveform, scs_freq, scs_mod, scs_mt,scs_ontime]).start()
    # 60 sec washout period
    threading.Timer(60, print, ['washing period for 60 sec']).start()
    threading.Timer(60, countdown, [60, 1]).start()
    time.sleep(120)

    return


def sr(sciatic_mt, status=1, sciatic_freq=0.33333, sciatic_pw=300, sciatic_ontime=10):
    #  8 intensities * (10+10) s =160s
    sr_intensity = [5, 10, 15, 20, 30, 40, 50, 75]

    t = threading.Timer(0, sample_trigger)
    t.start()
    t.join()

    for i in sr_intensity:
        sciatic_mod = i
        t = threading.Timer(0, sciatic, [status, sciatic_mt, sciatic_mod,
                                         sciatic_freq, sciatic_pw, sciatic_ontime])
        t.start()
        t.join()
        print('rest 10s')
        time.sleep(10)

    return


def windup(waveform, scs_freq, scs_mt, scs_mod, sciatic_mt, c_thres,
           sciatic_freq=1, status=1, sciatic_pw=2000):
    # 16 pulse 2ms 1Hz 3xC-threshold or 50xMT? (33s)
    #  recording length = 90s
    mp_bp = 0  # -1 for BP, 0 for mono phase pulse
    sciatic_mod = 3 * c_thres
    pulse = 16
    sciatic_ontime = pulse / sciatic_freq
    scs_ontime = pulse / sciatic_freq

    t1 = threading.Timer(0, sample_trigger)
    t1.start()
    t1.join()
    # start episode at t=0
    threading.Timer(0, wavegen, [waveform, scs_freq, scs_ontime, scs_mt, scs_mod]).start()
    threading.Timer(0, sciatic, [status, sciatic_mt, sciatic_mod, sciatic_freq,
                                 sciatic_pw, sciatic_ontime, mp_bp]).start()
    threading.Timer(0, scs, [waveform, scs_freq, scs_mod, scs_mt, scs_ontime]).start()
    threading.Timer(20, print, ['after_windup period will last 10 sec\n']).start()
    time.sleep(30)

    return


"""script for automated recording"""


def auto_char():
    print('map RF by 26g VF filament then\n'
          '(trigger from Channel 3 with 270 sec sampling length)\n')

    choice = input('START Char in LabChart then press ENTER to continue or 0 to exit\n')

    if choice == "0":
        print("now exit, no characterization, unspecified cell type")
    else:
        # start characterization process
        trigger_1 = threading.Timer(0, sample_trigger)
        trigger_1.start()
        trigger_1.join()
        threading.Timer(0, char).start()  # characterization 270s for Von Fray

        time.sleep(270)

        # determine cell type based on characterization
        cell_type = input('\ntype of cell? (0-Neither/1-NS/2-WDR) ')
        cell_type = int(cell_type)

        if cell_type == 0:
            io_char(cell_type, 0)
            print('Not typical NS/WDR neuron, SU recording will not start.')

        elif cell_type == 1 or 2:
            input('disconnect PowerLab AI 4 from AO 1')
            auto_sr(cell_type, **mt)

        else:
            print('none has specified, now exit auto mode, start auto_SR manually')

    return cell_type


def auto_sr(cell_type, sciatic_mt, **kw):

    cell = {1: 'NS', 2: 'WDR'}
    print('\nThis is a {0} neuron, S-R testing will start later.\n'
          'Save Char recording and START new SR recording in LabChart\n'
          '(trigger from Channel 3 with 160sec sampling length)'.format(cell[cell_type]))
    choice = input('Press ENTER when ready to proceed to S-R testing or 0 to exit\n')

    if choice == "0":
        print("now exit, no SR test, no c_threshold determined")
    else:
        # SR automation by calling sr function to determine c_threshold
        status = 1
        sciatic_freq = 0.3333333
        sciatic_pw = 300
        sciatic_ontime = 10

        sr(sciatic_mt, status, sciatic_freq, sciatic_pw, sciatic_ontime)

        # determine c_threhold based on SR result
        c_threshold = input('C-fiber threshold in sciatic MT mod(X)\n'
                            '5, 10, 15, 20, 30, 40, 50, 75x \n')
        c_threshold = int(c_threshold)
        io_char(cell_type, c_thres=c_threshold)

        choice = input('Save S-R testing recording and START new SU recording in LabChart\n'
                       '(trigger from Channel 3 with 120sec infinite sampling length)\n'
                       'Press ENTER when ready to proceed to SCS episodes or 0 to exit\n')
        if choice == "0":
            print("now exit, start auto_su manually")
        else:
            auto_su(c_thres=c_threshold, **mt)

    return c_threshold


def auto_su(bp50_mt, bp10k_mt, sciatic_mt, c_thres=20, sciatic_mod=50,
            scs_ontime=30, sciatic_freq=0.3333333, sciatic_pw=300, **kw):
    # timing clock: sciatic stim start at 0s, scs start at 20sec
    """
    para_dict = {0:((waveform_sel=waveform, scs_freq, scs_mt), scs_mod_sel, status_sel)}
    for i in roaster{'condition switch': [waveform, scs_freq, scs_mod, status]}:
           func body
           roaster[i][0 1 2 3] = waveform, scs_freq, scs_mod, status
    """

    para_change = input('any change to scs_ontime, sciatic_pw/freq/mod?\n'
                        'default value: 30 secs,   300 \u00b5s/0.3333333 Hz/50x (press ENTER if none)\n')

    if para_change == '':
        pass
    else:
        scs_ontime, sciatic_pw, sciatic_freq, sciatic_mod = map(int, para_change.split()[0:3])

    waveform_sel = [('BP_50', 50, bp50_mt), ('BP_10K', 10000, bp10k_mt)]
    scs_mod_sel = [20, 40, 80]
    status_sel = [0, 1]  # 25-75x MT, avg c-fiber thres=0.6-0.9mA 3x C-fiber=3mA ~30xMT

    para_comb = itertools.product(waveform_sel, scs_mod_sel, status_sel)
    para_list = list(para_comb)
    para_dict = dict(enumerate(para_list))

    n = len(para_dict)
    roster = list(range(n))

    indices = tuple(random.sample(roster, n))

    for i in indices:
        print('waveform/freq/mt={0} at {1}% MT with sciatic[{2}]'.format(*para_dict[i]))
        waveform = para_dict[i][0][0]
        scs_freq = int(para_dict[i][0][1])
        scs_mt = para_dict[i][0][2]
        scs_mod = int(para_dict[i][1])
        status = int(para_dict[i][2])
        scs_ontime = int(scs_ontime)  # 30s default
        num = i

        su(num, waveform, scs_freq, scs_ontime, scs_mt, scs_mod,
           status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw)

    wp = input('testing windup? (Y-1/N-0) \n')

    if wp == "1":
        input('finish SU recording, start new WINDUP recording in LabChart\n'
              '(trigger from Channel 3 with 30sec infinite sampling length)\n'
              'Press ENTER to start\n')
        auto_wp(c_thres, **mt)

    else:
        print('finish automation of SU recording without testing wind-up')

    return


def auto_wp(bp50_mt, bp10k_mt, sciatic_mt, c_thres=20, **kw):
    print('WINDUP recording starting')

    waveform_sel = [('BP_50', 50, bp50_mt), ('BP_10K', 10000, bp10k_mt)]
    scs_mod_sel = [80]

    para_comb = itertools.product(waveform_sel, scs_mod_sel)
    para_list = list(para_comb)
    para_list.append((('BP_50', 50, bp50_mt), 0))  # add windup episode w/o scs
    para_dict = dict(enumerate(para_list))

    n = len(para_dict)
    roster = list(range(n))

    for i in roster:
        print('waveform/freq/mt={0} at {1}% SCS MT during windup]'.format(*para_dict[i]))
        waveform = para_dict[i][0][0]
        scs_freq = int(para_dict[i][0][1])
        scs_mt = para_dict[i][0][2]
        scs_mod = int(para_dict[i][1])

        windup(waveform, scs_freq, scs_mt, scs_mod, sciatic_mt, c_thres)  # 16pulses - 30s
        time.sleep(210)

    return


"""define variables/calling functions"""
resources()
exp_rcd, mt = exp_recd_para()
neuron = auto_char()
c_thres = auto_sr(neuron, **mt)
auto_su(c_thres=c_thres, **mt)
auto_wp(c_thres=c_thres, **mt)







