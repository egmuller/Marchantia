# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:15:42 2022

@author: emuller
"""

## Clean up before script start 

for element in dir():

    if element[0:1] != "_":

        del globals()[element]

import gc
gc.collect()

## plotting stuff in jupyter
import matplotlib as mpl
mpl.use('TkAgg')
%matplotlib inline

COLOR = 'white'
COLOR2 = 'black'

mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
mpl.rcParams['axes.edgecolor'] = COLOR

mpl.rcParams["figure.facecolor"] = COLOR2
mpl.rcParams["axes.facecolor"] = COLOR2
mpl.rcParams["savefig.facecolor"] = COLOR2
mpl.rcParams['axes.facecolor'] = COLOR2


## to hide known warnings
import warnings
warnings.filterwarnings("ignore")

## my functions
import sys
sys.path.append(r'D:/Users/emuller/Documents/GitHubKraken/Marchantia/')
import VallapFunc as vf
from Main import BinarizeAndFitArea, BinarizeAndFitOsChoc
from GemmaeDetection import FindChipPos
import numpy as np

%run D:/Users/emuller/Documents/GitHubKraken/Marchantia/ExperimentList.py

BinarizeAndFitArea('221010_Ct',StackList221010_Ct,P221010_Ct,Scale221010,FPH221010,Delay221010,0.98,'H>','B',
                   debug = True, debugAll = False, debuglist = [3], HSVrange =[(15, 25, 40),(70, 150,210)])


