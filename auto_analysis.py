#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Auto-pilot SU recording for dorsal horn NS cell
 with Keithley 3390 waveform generator; Xcell3 amplifier; NIDAQ"""

import copy
import os
import threading
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import struct
import pandas as pd
from openpyxl import load_workbook
from scipy import signal, stats

"""
% matplotlib
inline
"""

def read_adibin(path):
    """read adibin with header structure
    <file_info>
    1. number og channel (n_channel), 2. sample per channel (sample_channel),
    3. time_channel=1 (time included), 4. data_type = double (1), float (2), short (3)
    <title>
    name of channels
    <channel_info>
    'unit', 'scale', 'range_high', 'range_low'
    <data_array> ch0:yime stamp; data in ch1-ch4
    {'ch0':array([time])...'ch4':array([data])}
    """

    file_path = str(path)
    with open(path, 'rb') as fid:
        adi_binary = fid.read()

    # header structure
    adi_file_format, adi_file_len = '=4sld5l2d4l', 68
    adi_channel_format, adi_channel_len = '=32s32s4d', 96

    # read file header
    adi_file_header = struct.unpack_from(adi_file_format, adi_binary, 0)

    # check file type
    magic4 = adi_file_header[0].decode('ascii')
    if magic4 != 'CFWB':
        raise ValueError('export file to adibin instead of adicht')

    # retrive recording information from header
    sec_tick = adi_file_header[2]  # 1/sec_tick=sampling freq; for 100k speed => 10 us
    n_channel, sample_channel, time_channel, data_ind = adi_file_header[10:14]
    data_type = ['d', 'f', 'h'][data_ind - 1]

    file_keys = ('sec_tick', 'n_channel', 'sample_channel', 'time_channel', 'data_type')
    file_values = (sec_tick, n_channel, sample_channel, time_channel, data_type)

    file_info = dict(zip(file_keys, file_values))  # general recording info

    # read 4 channel headers
    adi_channel_header = {}
    title = {}
    for x in range(0, 4):
        adi_channel_header['ch{0}'.format(x + 1)] = struct.unpack_from(
            adi_channel_format, adi_binary, adi_file_len + adi_channel_len * int(x))

        title['ch{0}'.format(x + 1)] = adi_channel_header['ch{0}'.format(x + 1)][0].decode('ascii').strip('\x00')

    unit = adi_channel_header['ch1'][1].decode('ascii').strip('\x00')
    scale = adi_channel_header['ch1'][2]
    range_high, range_low = [x / 10 ** 8 for x in adi_channel_header['ch1'][4:6]]
    channel_keys = 'unit', 'scale', 'range_high', 'range_low'
    channel_values = unit, scale, range_high, range_low
    channel_info = dict(zip(channel_keys, channel_values))  # general channel info

    # read recording data from 4 channels + time channel
    adi_data = struct.unpack_from(
        '={0}{1}'.format(((file_info['n_channel'] + 1) * file_info['sample_channel']), file_info['data_type']),
        adi_binary, adi_file_len + 4 * adi_channel_len)

    # parse data into 4+1 arrys corresponding to time stamp + recordings
    data_array = {}
    for x in range(file_info['n_channel'] + 1):
        # ch0 = time channel
        data_array['ch{0}'.format(x)] = np.array(adi_data[x::(file_info['n_channel'] + 1)], dtype=np.float)

    if len(data_array['ch0']) != file_info['sample_channel']:
        raise ValueError('sample number in array not match raw data')

    return file_info, channel_info, title, data_array


def get_ch_num(ch_title, title):
    ch_title = str(ch_title)
    ch_num = list(title.keys())[list(title.values()).index(ch_title)]

    return ch_num


def atf(mk_chan=None, rcd=None, mk_thres=0.03, atf_period=0.005, atf_freq=50, interval=None):
    # atf_period in ms
    # bp50_thres in chn2 in V
    atf_loc = []
    i = 0
    if interval == None:
        interval = file_info['sec_tick']
    while i < len(mk_chan) - int(atf_period / interval):
        if abs(mk_chan[i]) >= mk_thres:
            atf_loc.append(i)
            i += int(((1 / atf_freq) / interval) * 0.9)  # stim period(s)/interval(s) = points
        else:
            i += 1

    atf_clt = []

    for j in atf_loc:
        atf_temp = rcd[j:(j + int(atf_period / interval))]  # atf_period in points to get atf as list
        atf_clt.append(atf_temp)  # collection of atf

    atf_clt = np.array(atf_clt)
    atf_avg = [sum(atf) / len(atf) for atf in np.dstack(atf_clt)[0]]

    fig = display(atf_avg, xsize=6, ysize=2, linewidth=0.5)
    plt.show(fig)

    return atf_avg, atf_loc


def blank(mk_chan=None, rcd=None, mk_thres=0.03, atf_period=0.005, atf_freq=50, interval=None):
    # atf_period in ms
    # bp50_thres in chn2 in V
    if interval == None:
        interval = file_info['sec_tick']
    point = int(atf_period / interval)
    (atf_avg, atf_loc) = atf(mk_chan, rcd, mk_thres, atf_period, atf_freq)

    new_rcd = copy.copy(rcd)
    for i in range(len(atf_loc)):
        new_rcd[atf_loc[i]:atf_loc[i] + point] = [0] * point
    fig = display(rcd, xsize=15, ysize=3, ymin=-2, ymax=2, linewidth=0.1)
    fig = display(new_rcd, xsize=15, ysize=3, ymin=-2, ymax=2, linewidth=0.1)
    plt.show(fig)

    return new_rcd


def auto_atf(mk_chan=None, rcd=None, mk_thres=0.03, atf_period=0.005, atf_freq=50, interval=None):
    if interval == None:
        interval = file_info['sec_tick']
    point = int(atf_period / interval)
    (atf_avg, atf_loc) = atf(mk_chan, rcd, mk_thres, atf_period, atf_freq)
    choice = input('need adjust mk_thres={0} or atf_period={1}?(Y/N/exit)'.format(mk_thres, atf_period))

    while choice == 'Y':
        atf_avg, atf_loc = 0, 0
        new_mk_thres = float(input('atf_thres'))
        new_atf_period = float(input('atf_period(s)'))
        (atf_avg, atf_loc) = atf(mk_chan, rcd, mk_thres=new_mk_thres, atf_period=new_atf_period)
        print('out2', atf_loc)
        choice = input('need adjust mk_thres={0} or atf_period={1}?(Y/N/exit)'.format(new_mk_thres, new_atf_period))

    if choice == 'N':
        new_rcd = copy.copy(rcd)
        for i in range(len(atf_loc)):
            min_square = [float('inf')]
            offset = 0
            for k in range(-5, 9):
                square = [(m - n) ** 2 for m, n in zip(new_rcd[atf_loc[i] - k:atf_loc[i] - k + point], atf_avg)]
                if sum(square) < min_square[0]:
                    min_square[0] = sum(square)
                    offset = k
            ratio = max(new_rcd[atf_loc[i] - k:atf_loc[i] - k + point]) / max(atf_avg)
            atf_avg_temp = [t * ratio for t in atf_avg]
            new_rcd[atf_loc[i] - offset:atf_loc[i] - offset + point] = [m - n for m, n in
                                                                        zip(new_rcd[atf_loc[i] - offset:atf_loc[
                                                                                                            i] - offset + point],
                                                                            atf_avg_temp)]
        fig = display(rcd, xsize=15, ysize=3, ymin=-2, ymax=2, linewidth=0.1)
        fig = display(new_rcd, xsize=15, ysize=3, ymin=-2, ymax=2, linewidth=0.1)
        plt.show(fig)
    else:
        new_rcd = []
        pass

    return atf_avg, atf_loc, new_rcd


def display(data, edge=0.05, xsize=12, ysize=5, xmin=None, xmax=None, xlb=r'$\mu$s', ylb=r'$\mu$V',
            ymin=None, ymax=None, linewidth=0.5):
    if xmin is None:
        xmin = 0
    if xmax is None:
        xmax = len(data)

    if ymin is None:
        ymin = min(data) * 1.5
    if ymax is None:
        ymax = max(data) * 1.5

    plt.figure(figsize=(xsize, ysize))
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    blank = (xmax - xmin) * edge

    axes = plt.gca()
    axes.set_xlim([xmin - blank, xmax + blank])
    axes.set_ylim([ymin, ymax])
    # plt.xticks([60000, 80000, 100000, 120000], [0, 10, 20, 30])
    # plt.yticks([-1.2, 0, 1.2], [-100, 0, 100])

    sns.set_style("ticks")
    sns.despine()
    # axes.set_facecolor('w')

    plt.plot(data[0:-1], linewidth=linewidth)
    # plt.show()

    return


def peak(sci, rcd, noise=0.1, thres=1, limit=0.5, direction=1, height=2):
    """thres = signal threshold
       limit = sciatic threshold
       direction = 1: positive peak"""
    extrm_posit = []
    extrm_val = []
    local_extrm = thres
    i, j = 0, 0
    trend = 0
    index = list(range(len(sci)))
    sci_atfs = []  # position of sciatic stim

    while j + 100 < len(sci):  # finding sciatic stim location with minimum = limit
        if sci[j] > limit:
            sci_atfs.append(j)
            j += 100  # jump 100*10=1000us=1ms, sci stim pw=300us*2
        else:
            j += 1

    for m in sci_atfs:
        index[m - 100:m + 200] = [0] * 300  # omit area 1ms before and 3ms after sci stim

    while i + 50 < len(index):
        # ignore atf area (index[]=0) chasing up/down ignore temporary deflection<noise
        # mark the trend breach threshold (1) or remain sub-threshold (0)

        if abs(rcd[i]) > abs(local_extrm) and index[i] > 0:
            trend = 1  # find a trend breach threshold
            local_extrm = rcd[i]
            local_posit = i
        i += 1
        # when deflection>noise, mark last extrema as local peak, jump 50*10us=0.5ms, reset trend=0
        if abs(rcd[i]) < abs(local_extrm) - noise and trend == 1:
            extrm_posit.append(local_posit)
            extrm_val.append(local_extrm)
            i += 40
            trend = 0  # reset trend
            local_extrm = thres  # reset local_extrm value
    # validate peaks by height

    for x, y in enumerate(extrm_posit):
        if abs(min(rcd[y - 30:y + 30]) - extrm_val[x]) < height and extrm_val[x] > 0:
            extrm_val.pop(x)
            extrm_posit.pop(x)
        elif abs(max(rcd[y - 30:y + 30]) - extrm_val[x]) < height and extrm_val[x] < 0:
            extrm_val.pop(x)
            extrm_posit.pop(x)

    return extrm_posit, extrm_val, sci_atfs


def scaling(rcd, one=0.6, exp=1.6, limit=1.5):
    scale_rcd = []

    for i in rcd:
        if i >= 0:
            scale_rcd.append((i / one) ** exp)
        elif i < 0:
            scale_rcd.append((((i * -1) / one) ** exp) * (-1))
    for index, i in enumerate(scale_rcd):
        if i > limit:
            scale_rcd[index] = limit
        elif i < limit * -1:
            scale_rcd[index] = limit * -1
        else:
            pass

    return np.array(scale_rcd)


def elliptic(wave):
    data = wave
    ps = np.abs(np.fft.fft(data))
    ps_length = len(ps)
    ps_half = ps[0:int(ps_length * 0.5)]
    print(data.size)

    time_step = 1 / 100000
    freqs = np.fft.fftfreq(data.size, time_step)
    freqs_half = freqs[0:int(ps_length * 0.5)]
    idx = np.argsort(freqs_half)
    fig = plt.figure(figsize=(30, 10))
    plt.axis([0, 5000, 0, 5000])
    plt.plot(freqs_half[idx], ps_half[idx])
    # plt.savefig('c:/users/stevi/desktop/TB_analyze/output.png')
    plt.show(fig)
    return


def sc_temp(data_array):
    """get data from ch3: 'Trigger/Temp'"""
    sc_temp = data_array[get_ch_num('Trigger/Temp', title=title)]
    avg_sc_temp = (np.mean(sc_temp, axis=0)) * 1000
    sd_sc_temp = np.std(sc_temp, ddof=1)
    print('SC Temperature (mean \u00B1 SD)= {0:.3f} \u00B1 {1:.3f}'.format(avg_sc_temp, sd_sc_temp))

    return avg_sc_temp, sd_sc_temp


def sci_atf(sci, sci_thres):
    sci_atfs = []  # position of sciatic stim
    j = 0
    while j + 100 < len(sci):  # finding sciatic stim location with minimum = limit
        if sci[j] > sci_thres:
            sci_atfs.append(j)
            j += 10000  # jump 10000*10=100000us=100ms, sci stim pw=300us*2
        else:
            j += 1

    return sci_atfs


def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    sos = signal.butter(order, [low, high], btype='bandstop', output='sos')
    y = signal.sosfilt(sos, data)
    return y, sos


def PSTH(data, xsize=12, ysize=5, bin_width=500, xmax=None, ymax=None):
    xlb = r'bin (5ms)'
    ylb = r'Spikes'
    edge = 0.05
    xmin, ymin = 0, 0

    fig = plt.figure(figsize=(xsize, ysize))
    blank = (xmax - xmin) * edge

    axes = plt.gca()
    axes.set_xlabel(xlb)
    axes.set_ylabel(ylb)
    axes.set_xlim([xmin - blank, xmax + blank])
    axes.set_ylim([ymin, ymax])
    # axes.set_xticks([20000, 40000, 60000, 80000], [200, 400, 600, 800])
    # plt.yticks([-1.2, 0, 1.2], [-100, 0, 100])

    sns.set_style("ticks")
    sns.despine()
    # axes.set_facecolor('w')

    bins = np.arange(0, 80000, bin_width)
    plt.hist(data, bins=bins, alpha=0.5)
    boundary = [200, 3000, 9000, 30000]
    plt.vlines(x=boundary, ymin=ymin, ymax=ymax, colors='r')

    fig.savefig('c:/users/stevi/desktop/TB_analyze/PSTH_{0}_{1}_{2}.pdf'.format(hd[0], hd[1], hd[2]),
                bbox='tight', facecolor='w', transparent=False)
    return


def avg_amp(rcd):
    avg_amp = (max(rcd) - min(rcd)) / 2
    ymin = -avg_amp
    ymax = avg_amp
    return ymin, ymax