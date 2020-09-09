#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
            '<#rat, #neuron, depth, angle, Z of electrode(M)>\n'
        ).split()
        exp_rcd_val.append(local_time)
        loc_exp_rcd = dict(zip(exp_rcd_key, exp_rcd_val))

        mt_key = ['sciatic_mt', 'bp50_mt', 'bp1k_mt', 'bp1_2k_mt', 'bp10k_mt']
        mt_val = input('\nInput MT, separated by space, for:\n'
                       '<(1)Sciatic, (2)BP50, (3)BP1K, (4)BP1_2K, (5)BP10K in Vpp>\n').split()
        mt_val[:] = list(map(float, mt_val[:]))
        loc_mt = dict(zip(mt_key, mt_val))

    except ValueError:
        print('must input arg separated by space')

    else:
        print('\nExperiment record of <{5}>\n'
              '\nRat[{0}]/neuron[{1}] at depth:{2}\u00b5m/angle:{3}\u00b0 with Z:{4} electrode\n'
              .format(*loc_exp_rcd.values()))
        print('sciatic_mt={0}, bp50_mt={1}, bp1k_mt={2}, bp1_2k_mt={3}, bp10k_mt={4} (Vpp)\n'.format(*loc_mt.values()))

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
    curr_dir_path = os.path.expanduser('~')
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
    curr_dir_path = os.path.expanduser('~')
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

        print('\nsampling start...')

        task.start()
        task.write([True, False])
        task.stop()

    return


def char(char_list):
    # total 250=12*20sec+10s at the end

    print('\ncharacterization start')

    for i in char_list:
        print('\nwait 10sec, prepare for {0}'.format(i))
        countdown(10, 3)
        print('start {0} for 10 secs'.format(i))
        countdown(10, 1)

    time.sleep(10)

    print('\ncharacterization is ending')

    return


def wavegen(waveform, scs_freq, scs_ontime, scs_mt, scs_mod):
    # Keithley 3390 configuration to be triggered/gated by NI-DAQ 6216 BNC with logic signal

    rm = visa.ResourceManager()  # assign NI backend as resource manager
    keithley = rm.open_resource('usb0::0x05E6::0x3390::1425019::INSTR')  # open keithley
    keithley.timeout = None
    keithley.write('*rst; *cls')  # reset to factory default, clear status command

    # define arbitrary waveform, burst/trigger, frequency/amplitude
    if waveform == 'DC':
        keithley.write('function DC')
        keithley.write('voltage:offset 0')
    else:
        keithley.write('function user')
        keithley.write('function:user {0}'.format(str(waveform)))

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
        keithley.write('voltage {0}'.format(str(scs_mt * scs_mod / 100)))  # scs_mod/100 = %
        keithley.write('voltage:offset 0')

        keithley.write('burst:state on')

    keithley.close()  # close the instrument handle session

    return


def sciatic(status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw, sciatic_ontime=60, mp_bp=-1):
    # time unit = 100us; SI from 100us/V => 1mA/V, bp=-1 mp=1
    """
    def time_pad_gen(status, freq, pw, on_time=60):
        time_pad_unit = [amp * status] * 300 + [0] * 700  # 1s template
        time_pad = time_pad_unit * 60  # 60s time pad
        return time_pad
    """

    switch = {1: 'on', 0: 'off'}

    num_si = 1.0  # currently using Caputron

    amp = status * sciatic_mt * sciatic_mod / 10  # 10(mA/V)=[1*1(100uA/V)*100(mod)]/10
    if amp > 9.96:  # max amp = 10 for A-M2000 5Vpp*2, Caputron 10Vpp*1
        amp = 9.96
    output_comm = round(amp / num_si, 2)  # Caputron:output=amp, A-M2200 output=amp/2

    step = 100  # us
    pad = (1/sciatic_freq) * (10 ** 6) / step  # length(1/freq) in 100us step
    sample_per_sec = float((10 ** 6) / step)  # specify sample per second for NI
    scale_wave = np.zeros(int(pad))
    scale_pw = math.floor(sciatic_pw/step)  # covert pw from 300us to 3 points in pad
    scale_wave[:scale_pw] = output_comm
    scale_wave[scale_pw:2*scale_pw] = mp_bp * output_comm
    scale_wave[-1] = 0  # force zero at the end

    with daq.Task() as task:
        # CONTINUOUS output within sciatic_ontime with designated waveform pad
        task.ao_channels.add_ao_voltage_chan('Dev1/ao0', min_val=-10.0, max_val=10.0
                                             , units=cons.VoltageUnits.VOLTS)
        task.timing.cfg_samp_clk_timing(sample_per_sec, sample_mode=cons.AcquisitionType.CONTINUOUS
                                        , samps_per_chan=len(scale_wave))
        task.out_stream.output_buf_size = len(scale_wave)  # buffer size:= sample size

        print('Sciatic stimulation is {0} (amp = {1}mA) for {2}sec\n'
              .format(switch[status], amp, sciatic_ontime))
        # num_sci_stim = round(sciatic_ontime * sciatic_freq)
        task.write(scale_wave, auto_start=False)

<<<<<<< HEAD
        # beep(3)
=======
>>>>>>> origin/master
        task.start()
        time.sleep(sciatic_ontime*1.01)  # let the sciatic stim waveform finish
        task.stop()

    with daq.Task() as task:
        # finite output 0 to compulsorily reset the output
        task.ao_channels.add_ao_voltage_chan('Dev1/ao0', min_val=-10.0, max_val=10.0
                                             , units=cons.VoltageUnits.VOLTS)
        task.timing.cfg_samp_clk_timing(sample_per_sec, sample_mode=cons.AcquisitionType.FINITE
                                        , samps_per_chan=2)
        task.out_stream.output_buf_size = 2
        task.write([0, 0], auto_start=True)
        task.wait_until_done(0.5)
        task.stop()

<<<<<<< HEAD
    # beep(1)

=======
>>>>>>> origin/master
    return


def scs(waveform, scs_freq, scs_mod, scs_mt, scs_ontime):

    si = 2.0  # number of si, currently 2x A-M 2200
    scs_amp = scs_mt * si

    rm = visa.ResourceManager()  # assign NI backend as resource manager
    keithley = rm.open_resource('usb0::0x05E6::0x3390::1425019::INSTR')  # open keithley
    keithley.write('OUTput ON')  # output on at burst mode, wait for trigger
<<<<<<< HEAD
    si = 2.0
    scs_amp = scs_mt * si
=======

>>>>>>> origin/master
    with daq.Task() as task:
        task.do_channels.add_do_chan('Dev1/port1/line0')  # PFI 0 /P1.0

        print('\n{0} stimulation at {1}Hz, {2}% MT: {3}*{4}SI={5}\n'
              .format(waveform, scs_freq, scs_mod, scs_mt, si, scs_amp))
        # task.start()
        task.write([True, False], auto_start=True)
        countdown(scs_ontime, beep_freq=1)
        keithley.write('OUTput OFF')
        task.stop()

    return


<<<<<<< HEAD
def sr(sciatic_mt, status=1, sciatic_freq=0.5, sciatic_pw=300, sciatic_ontime=10):
    #  11 intensities * (10+10) s =220s+20s spare =240s
    sr_intensity = [0.5, 1, 5, 10, 15, 20, 30, 40, 50, 75]
=======
def sr(sr_intensity, sciatic_mt, status=1, sciatic_freq=0.5, sciatic_pw=300, sciatic_ontime=10):
    #  11 intensities * (10+10) s =200s+20s spare =240s
>>>>>>> origin/master

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


def su(num, waveform, scs_freq, scs_ontime, scs_mt, scs_mod,
       status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw):

    io_su(num, waveform, scs_freq, scs_ontime, scs_mt, scs_mod,
          status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw)
    wavegen(waveform, scs_freq, scs_ontime, scs_mt, scs_mod)

    # check inputs/raise error before exec
    # trigger sampling at t=0 for 60 sec
    t = threading.Timer(0, sample_trigger)
    t.start()
    t.join()
    # start episode at t=0
    threading.Timer(0, sciatic, [status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw]).start()
    # threading.Timer(0, wavegen, [waveform, scs_freq, scs_ontime, scs_mt, scs_mod]).start()
    # start scs stim at t=15, for 30s
    threading.Timer(15, scs, [waveform, scs_freq, scs_mod, scs_mt, scs_ontime]).start()
    # 60 sec washout period
    threading.Timer(60, print, ['washing period for 60 sec']).start()
    threading.Timer(60, countdown, [60, 1]).start()
    time.sleep(120)

    return


def windup(waveform, scs_freq, scs_mt, scs_mod, sciatic_mt, c_thres,
           sciatic_freq=1, status=1, sciatic_pw=2000):
    # 16 pulse 2ms 1Hz 3xC-threshold or 50xMT? (33s)
    #  recording length = 90s
    mp_bp = -1  # -1 for BP, 0 for mono phase pulse
    sciatic_mod = 3 * c_thres
    pulse = 16
    sciatic_ontime = pulse / sciatic_freq
    scs_ontime = 30  # short SCS

    wavegen(waveform, scs_freq, scs_ontime, scs_mt, scs_mod)

    beep(3)  # latency between wevegen and scs

    wavegen(waveform, scs_freq, scs_ontime, scs_mt, scs_mod)

    t1 = threading.Timer(0, sample_trigger)
    t1.start()
    t1.join()
    # start episode at t=0
<<<<<<< HEAD
    # threading.Timer(0, wavegen, [waveform, scs_freq, scs_ontime, scs_mt, scs_mod]).start()
    threading.Timer(0, sciatic, [status, sciatic_mt, sciatic_mod, sciatic_freq,
                                 sciatic_pw, sciatic_ontime, mp_bp]).start()
=======

>>>>>>> origin/master
    threading.Timer(0, scs, [waveform, scs_freq, scs_mod, scs_mt, scs_ontime]).start()
    threading.Timer(30, sciatic, [status, sciatic_mt, sciatic_mod, sciatic_freq,
                                 sciatic_pw, sciatic_ontime, mp_bp]).start()
    threading.Timer(50, print, ['after_windup period will last 10 sec\n']).start()
    time.sleep(30)
    print('after windup for 210s')

    return


"""script for automated recording"""


def auto_char():
    print('map RF by 180g VF filament then\n'
          '(trigger from Channel 3 with 250 sec sampling length)\n')

    choice = input('\nSTART Char in LabChart then press ENTER to continue or 0 to exit')

    if choice == "0":
        print("\nnow exit, no characterization, no cell_type determined")
        cell_type = None
    else:
        # start characterization process
        trigger_1 = threading.Timer(0, sample_trigger)
        trigger_1.start()
        trigger_1.join()
        threading.Timer(0, char, [char_list]).start()  # characterization 210s for Von Fray

        time.sleep(250)

        # determine cell type based on characterization
        cell_type = input('\ntype of cell? (0-Neither/1-NS/2-WDR) \n')
        cell_type = int(cell_type)

        print('\nsave Char recording and START new SR recording in LabChart')

    return cell_type


def auto_sr(cell_type, sciatic_mt, sciatic_freq=0.5, sciatic_pw=300, **kw):

    if cell_type is None:
        cell_type = input('\nPlease specify cell_type now (0-Neither/1-NS/2-WDR)\n')
        cell_type = int(cell_type)

    cell = {1: 'NS', 2: 'WDR'}
    status = 1
    sciatic_ontime = 10

    choice = input('\nDisconnect PowerLab AI 4 from AO 1 \n'
                   'This is a {0} neuron, S-R testing will start later at {1}Hz//{2}uS.\n'
                   '(trigger from Channel 3 with 240sec sampling length)\n'
                   'Press ENTER when ready to proceed to S-R testing or 0 to exit\n'
                   .format(cell[cell_type], sciatic_freq, sciatic_pw))

    if choice == "0":
        print("now exit, no SR test, no c_threshold determined\n")
        c_threshold = None
    else:
        # SR automation by calling sr function to determine c_threshold
        sr(sr_intensity, sciatic_mt, status, sciatic_freq, sciatic_pw, sciatic_ontime)

        # determine c_threhold based on SR result
        c_threshold = input('\nC-fiber threshold in sciatic MT mod(X)\n'
                            '5, 10, 15, 20, 25, 30, 35, 40, 50, 75x \n')
        c_threshold = int(c_threshold)
        io_char(cell_type, c_thres=c_threshold)

        print('Save S-R testing recording and START new SU recording in LabChart\n')

    return c_threshold


def auto_su(bp50_mt, bp1k_mt, bp1_2k_mt, bp10k_mt, sciatic_mt, c_thres=30,
            scs_ontime=30, sciatic_freq=0.3333333, sciatic_pw=300, **kw):
    # timing clock: sciatic stim start at 0s, scs start at 15sec
    """
    para_dict = {0:((waveform_sel=waveform, scs_freq, scs_mt), scs_mod_sel, status_sel)}
    for i in roaster{'condition switch': [waveform, scs_freq, scs_mod, status]}:
           func body
           roaster[i][0 1 2 3] = waveform, scs_freq, scs_mod, status
    """
    print('Initiating SU recording \n'
          '(trigger from Channel 3 with 120sec infinite sampling length)\n'
          'Press ENTER when ready to proceed to SCS episodes or 0 to exit\n')

    if c_thres is None:
        c_thres = input('\nPlease specify c_threshold (5, 10, 15, 20, 25, 30, 35, 40, 50, 75)\n')
        c_thres = int(c_thres)

    sciatic_mod = 2 * c_thres

    para_change = input('\nany change to scs_ontime, sciatic_pw/freq/mod?\n'
                        'default value: 30 secs,   300 \u00b5s/0.33 Hz/2xC-thres (press ENTER if none)\n'
                        'Press "0" to exit or ENTER to continue\n')

    if para_change == '':
        pass
    elif para_change == '0':
        print("now exit, no SU recording\n")
    else:
        scs_ontime, sciatic_pw, sciatic_freq, sciatic_mod = map(int, para_change.split()[0:3])

    waveform_sel = [('BP50Hz_K', 50, bp50_mt), ('BP10KHz_K', 10000, bp10k_mt), ('BP1_2KHZ', 1200, bp1_2k_mt)]
    scs_mod_sel = [40, 80]
    status_sel = [0, 1]  # MT=0.50-100uA, avg c-fiber thres=0.6-0.9mA; 3x C-fiber=1.8-2.7mA ~30xMT

    para_comb = itertools.product(waveform_sel, scs_mod_sel, status_sel)
    para_list = list(para_comb)
    para_dict = dict(enumerate(para_list))

    n = len(para_dict)
    roster = list(range(n))

    indices = tuple(random.sample(roster, n))

    for i in indices:
        print('waveform/freq/mt={0} at {1}% MT with sciatic[{2}]\n'.format(*para_dict[i]))
        waveform = para_dict[i][0][0]
        scs_freq = int(para_dict[i][0][1])
        scs_mt = para_dict[i][0][2]
        scs_mod = int(para_dict[i][1])
        status = int(para_dict[i][2])
        scs_ontime = int(scs_ontime)  # 30s default
        num = i

        su(num, waveform, scs_freq, scs_ontime, scs_mt, scs_mod,
           status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw)

    print('save SU recording, start new WINDUP recording in LabChart\n')


    return


def auto_wp(bp50_mt, bp1k_mt, bp1_2k_mt, bp10k_mt, sciatic_mt, c_thres=30, **kw):
    # 3 SCS conditions: DC, 50Hz, 10KHz at 40% mt; 3*4m=12m
    if c_thres is None:
        c_thres = input('\nPlease specify c_threshold (5, 10, 15, 20, 25, 30, 35, 40, 50, 75)\n')
        c_thres = int(c_thres)

    wp = input('\n(trigger from Channel 3 with 30sec infinite sampling length)\n'
               'Press ENTER to start or "0" to exit\n')

    if wp == '0':
        print('WINDUP aborted\n')
    else:
        print('WINDUP recording initiating\n')

        waveform_sel = [('BP50Hz_K', 50, bp50_mt), ('BP10KHz_K', 10000, bp10k_mt),('BP1_2KHZ', 1200, bp1_2k_mt)]
        scs_mod_sel = [40, 80]

        para_comb = itertools.product(waveform_sel, scs_mod_sel)
        para_list = list(para_comb)
        para_list.insert(0, (('DC', 0, 0), 0))  # add windup episode w/o scs
        para_dict = dict(enumerate(para_list))

        n = len(para_dict)
        roster = list(range(n))

        for i in roster:
            print('waveform/freq/mt={0} at {1}% SCS MT during windup]\n'.format(*para_dict[i]))
            waveform = para_dict[i][0][0]
            scs_freq = int(para_dict[i][0][1])
            scs_mt = para_dict[i][0][2]
            scs_mod = int(para_dict[i][1])

            windup(waveform, scs_freq, scs_mt, scs_mod, sciatic_mt, c_thres)  # 16pulses - 30s
            time.sleep(150)
        print('save WINDUP recording and start new PRL-SCS recording in LabChart\n')

    return


def long_scs(sciatic_mt, bp50_mt, bp1k_mt, bp1_2k_mt, bp10k_mt, scs_freq=10000, c_thres=30, scs_mod=40, **kw):
    # 50Hz or 10KHz at 40% mt for 30min prolong stimulation
    # test with 1) 0.33Hz/2x C-thres 30s for c-fiber; 2)wind-up protocol: 1Hz, 3x C-thres for 16s
    # each trigger record for 180s = 0.1*total length

    # SCS
    scs_ontime = int(1500)

    # sciatic
    status = 1
    mp_bp = -1

    # c-fiber
    sciatic_mod = 2 * c_thres
    sciatic_freq = 0.333
    sciatic_pw = 300
    sciatic_ontime = (scs_ontime / 60) + 1  # 31s

    # wind-up
    wp_mod = 3 * c_thres
    wp_freq = 1
    wp_pw = 2000
    wp_ontime = (scs_ontime / 120) + 1  # 16s

    scs_para = {'10000': (bp10k_mt,'BP10KHz_K'), '50': (bp50_mt, 'BP50Hz_K'),
                '1000': (bp1k_mt, 'BP1K_24u_K'), '1200': (bp1_2k_mt, 'BP1_2KHz')}

    scs_mt = scs_para[str(scs_freq)][0]
    waveform = scs_para[str(scs_freq)][1]

    """
    if scs_freq == 10000:
        scs_mt = bp10k_mt
        waveform = 'BP10KHz_K'
    elif scs_freq == 50:
        scs_mt = bp50_mt
        waveform = 'BP50Hz_K'
    elif scs_freq == 1000:
        scs_mt = bp1k_mt
        waveform = 'BP1K_24u_K'
    elif scs_freq == 1200:
        scs_mt = bp1_2k_mt
        waveform = 'BP1_2KHz'
    """

    choice = input ('\nstart PRL recording? check coverter of PhysioTemp probe\n'
                    'Enter to continue or "0" to exit\n')

    if choice == '0':
        print('exit PRL recording now')
    else:
        # register to Keithley
        wavegen(waveform, scs_freq, scs_ontime, scs_mt, scs_mod)

        # pre-scs baseline 30s c-fiber + 60s blank + 15s wp +195 rest = 300 total
        t0 = threading.Timer(0, sample_trigger)
        t0.start()
        t0.join()
        threading.Timer(0, print, ['pre-SCS session']).start()
        threading.Timer(0, sciatic,
                        [status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw, sciatic_ontime, mp_bp]).start()
        threading.Timer(0.04 * scs_ontime, sciatic, [status, sciatic_mt, wp_mod, wp_freq, wp_pw, wp_ontime, mp_bp]).start()

        # prolong scs session after 0.2*scs_ontime=300s
        t1 = threading.Timer(0.2 * scs_ontime, sample_trigger)
        t1.start()
        t1.join()
        threading.Timer(0, print, ['initiate prolong SCS session']).start()
        threading.Timer(0, scs, [waveform, scs_freq, scs_mod, scs_mt, scs_ontime]).start()
        threading.Timer(0, sciatic,
                        [status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw, sciatic_ontime, mp_bp]).start()
        threading.Timer(0.04 * scs_ontime, sciatic, [status, sciatic_mt, wp_mod, wp_freq, wp_pw, wp_ontime, mp_bp]).start()

        t2 = threading.Timer(0.25 * scs_ontime, sample_trigger)
        t2.start()
        t2.join()
        threading.Timer(0, sciatic,
                        [status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw, sciatic_ontime, mp_bp]).start()
        threading.Timer(0.04 * scs_ontime, sciatic, [status, sciatic_mt, wp_mod, wp_freq, wp_pw, wp_ontime, mp_bp]).start()

        t3 = threading.Timer(0.25 * scs_ontime, sample_trigger)
        t3.start()
        t3.join()
        threading.Timer(0, sciatic,
                        [status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw, sciatic_ontime, mp_bp]).start()
        threading.Timer(0.04 * scs_ontime, sciatic, [status, sciatic_mt, wp_mod, wp_freq, wp_pw, wp_ontime, mp_bp]).start()

        t4 = threading.Timer(0.25 * scs_ontime, sample_trigger)
        t4.start()
        t4.join()
        threading.Timer(0, sciatic,
                        [status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw, sciatic_ontime, mp_bp]).start()
        threading.Timer(0.04 * scs_ontime, sciatic, [status, sciatic_mt, wp_mod, wp_freq, wp_pw, wp_ontime, mp_bp]).start()
        threading.Timer(0.1 * scs_ontime, print, ['end of prolong SCS session, post-session starting soon']).start()

        # post-scs session; 0.2*scs_ontime after=300s
        t5 = threading.Timer(0.25 * scs_ontime+10, sample_trigger)
        t5.start()
        t5.join()
        threading.Timer(0, print, ['post-SCS session']).start()
        threading.Timer(0, sciatic,
                        [status, sciatic_mt, sciatic_mod, sciatic_freq, sciatic_pw, sciatic_ontime, mp_bp]).start()
        threading.Timer(0.04 * scs_ontime, sciatic, [status, sciatic_mt, wp_mod, wp_freq, wp_pw, wp_ontime, mp_bp]).start()
        threading.Timer(0.1 * scs_ontime, print, ['end of post-SCS session']).start()

    return


"""define variables/calling functions"""
resources()
exp_rcd, mt = exp_recd_para()
neuron = auto_char()
c_threshold = auto_sr(neuron, **mt)
auto_su(c_thres=c_threshold, **mt)
auto_wp(c_thres=c_threshold, **mt)
long_scs(c_thres=c_threshold, **mt)






