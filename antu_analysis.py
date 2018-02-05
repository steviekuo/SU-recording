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


def read_adicht():

    f_path = input('enter absolute path of file')

    ch1 = []
    ch2 = []
    ch3 = []
    ch4 = []

    with open(f_path, 'r') as fid:
        for line in fid.readline():
            ch1.append(line.split()[0])
            ch2.append(line.split()[1])
            ch3.append(line.split()[2])
            ch4.append(line.split()[3])

    return ch1, ch2, ch3, ch4


def read_adibin():

    f_path = input('enter absolute path of file')

    ch1 = []
    ch2 = []
    ch3 = []
    ch4 = []

    with open(f_path, 'rb') as fid:
        for line in fid.readline():
            ch1.append(line.split()[0])
            ch2.append(line.split()[1])
            ch3.append(line.split()[2])
            ch4.append(line.split()[3])

    return ch1, ch2, ch3, ch4