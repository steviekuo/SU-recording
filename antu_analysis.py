#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import os
import time
import datetime
import random
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# % matplotlib inline


def read_adibin(path):
    f_path = path

    tik = []
    rcd = []
    scs = []
    trg = []
    sci = []

    with open(f_path, 'r') as fid:
        lines = fid.readlines()
        interval = float(lines[0].split('\t')[1].rstrip(' s\n'))

        for line in lines[6:]:
            tik.append(float(line.split()[0]))
            rcd.append(float(line.split()[1]))
            scs.append(float(line.split()[2]))
            trg.append(float(line.split()[3]))
            sci.append(float(line.split()[4]))

    display(rcd, xsize=12, ysize=6, ymin=-1, ymax=1, linewidth=0.3)

    return interval, tik, rcd, scs, trg, sci


def atf(interval, scs, rcd, bp50_thres=0.03, atf_period=0.003, atf_freq=50):
    # atf_period in ms
    # bp50_thres in chn2 in V
    loc_atf = []
    i = 0
    while i < len(scs):
        if scs[i] >= bp50_thres:
            loc_atf.append(i)
            i += int((1 / atf_freq) / interval)  # pulse width(s)/interval(s) = points
        else:
            i += 1

    atf_clt = []

    for j in loc_atf:
        atf_temp = rcd[j:(j + int(atf_period / interval))]  # 5ms/interval both in s
        atf_clt.append(atf_temp)
    atf_avg = [sum(atf) / len(atf) for atf in zip(*atf_clt)]

    display(atf_avg, xsize=6, ysize=2, ymin=-1, ymax=1, linewidth=0.5)

    return atf_avg, loc_atf


def auto_atf(interval, bp50_thres=0.03, atf_period=0.002, atf_freq=50):
    point = int(atf_period / interval)
    (atf_avg, loc_atf) = atf(interval, scs, rcd, bp50_thres, atf_period, atf_freq)
    choice = input('need adjust bp50_thres={0} or atf_period={1}?(Y/N/exit)'.format(bp50_thres, atf_period))

    while choice == 'Y':
        new_bp50_thres = float(input('bp50_thres'))
        new_atf_period = float(input('atf_period(s)'))
        (atf_avg, loc_atf) = atf(interval, scs, rcd, bp50_thres=new_bp50_thres, atf_period=new_atf_period)
        choice = input('need adjust bp50_thres={0} or atf_period={1}?(Y/N/exit)'.format(new_bp50_thres, new_atf_period))

    if choice == 'N':
        new_rcd = copy.copy(rcd)
        for i in loc_atf:
            offset = atf_avg.index(max(atf_avg)) - rcd[i:i + point].index(max(rcd[i:i + point]))
            new_rcd[i - offset:i - offset + point] = [m - n for m, n in
                                                      zip(new_rcd[i - offset:i - offset + point], atf_avg)]
        display(new_rcd, xsize=15, ysize=3, ymin=-1, ymax=1, linewidth=0.3)

    else:
        pass

    return atf_avg, loc_atf, new_rcd


def display(data, edge=0.05, xsize=18, ysize=6, xmin=None, xmax=None, xlb='s', ylb='uV', linewidth=0.3,
            ymin=-2, ymax=2):
    if xmin is None:
        xmin = 0
    if xmax is None:
        xmax = len(data)

    plt.figure(figsize=(xsize, ysize))
    plt.xlabel(xlb)
    plt.ylabel(ylb)
    blank = (xmax - xmin) * edge

    axes = plt.gca()
    axes.set_xlim([xmin - blank, xmax + blank])
    axes.set_ylim([ymin, ymax])
    # plt.xticks([xmin, 0, xmax], [-100, 0, 100])
    # plt.yticks([ymin, 0, ymax], [-100, 0, 100])

    plt.plot(data[0:-1], linewidth=linewidth)
    plt.show()
    plt.savefig('c:/users/stevi/desktop/output.png')

    return


def peak(sci, rcd, noise=0.1, thres=0.5, limit=1):
    max_posit = []
    min_posit = []
    max_val = []
    min_val = []
    local_max = thres
    local_min = thres * -1
    i, j = 0, 0
    trend = 0
    index = list(range(len(sci)))
    sci_atf = []  # position of sciatic stim

    while j + 100 < len(sci):
        if sci[j] > limit:
            sci_atf.append(j)
            j += 100  # jump 100810=1000us=1ms, sci stim pw=300us*2
        else:
            j += 1

    for m in sci_atf:
        index[m - 100:m + 900] = [0] * 1000  # omit area 0.5ms before and 9.5ms after sci stim

    while i + 50 < len(index):
        # ignore atf area (index[]=0) chasing up/down ignore temporary deflection<noise
        # mark the trend as up/down (1, -1) or remain sub-threshold (0)
        if index[i] > 0 and rcd[index[i]] > local_max:
            trend = 1  # up for max
            local_max = rcd[i]
            local_posit = i
        elif index[i] > 0 and rcd[index[i]] < local_min:
            trend = -1  # down for min
            local_min = rcd[i]
            local_posit = i

        i += 1

        # when deflection>noise, mark last extrema as local peak, jump 50*10us=0.5ms, reset trend=0
        if trend == 1:
            if index[i] > 0 and rcd[i] < local_max - noise:
                max_posit.append(local_posit)
                max_val.append(local_max)
                i += 10
                trend = 0
                local_max = thres
        elif trend == -1:
            if index[i] > 0 and rcd[i] > local_min + noise:
                min_posit.append(local_posit)
                min_val.append(local_min)
                i += 10
                trend = 0
                local_min = thres * -1

    return max_posit, min_posit, max_val, min_val


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

    return scale_rcd


# read data into python
(interval, tik, rcd, scs, trg, sci) = read_adibin('c:/users/stevi/desktop/su_10k_80mt_suppress.txt')
display(rcd, xsize=12, ysize=6)

# remove artifact from 50Hz SCS
(atf_avg, loc_atf, new_rcd)=auto_atf(interval, bp50_thres=0.03, atf_period=0.002, atf_freq=50)

# scaling data
scale_rcd = scaling(rcd, one=0.6, exp=1.6, limit=1.5)
display(scale_rcd, xsize=12, ysize=5, ymin=-1, ymax=1, linewidth=0.5)

# peak detection and plot
(max_posit, min_posit, max_val, min_val) = peak(sci, scale_rcd, noise=0.2, thres=0.25, limit=1)
display(scale_rcd, xsize=18, ysize=6, linewidth=0.3)
plt.scatter(max_posit, max_val, cmap='b')
plt.scatter(min_posit, min_val, cmap='r')

