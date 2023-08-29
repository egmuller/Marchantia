# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 2022

@author: Valentin Laplaud

List of experiments

"""

import numpy as np



# Ximena's experiments

#CHIP

Scale_Ctrl = 1.94 # Spatial scale (µm/px) 
FPH170823 = 0 # Frames per hour
Delay = np.asarray([85+i*30 for i in range(0, 20)]+ [85+10*60+i*10 for i in range(0, 6)] + [85+10*60+60+i*30 for i in range(0,39)])


#Weight

 

#FPH230823 = 6 # Frames per hour
#Delay = 0 # Delay between imbibition and experiment start, in min
Scale_Weight0 = 0.92 # Spatial scale (µm/px)
Scale_Weight2 = 0.90 # Spatial scale (µm/px) 
Scale_Weight4 = 0.92

#Where to save files

P230817_0 = r'F:/Chip/Experiment_2/0M_Gemma'
P230817_200 = r'F:/Chip/Experiment_2/200M_Gemma'
#P230817_150_V2 = r'F:/Chip/Chamber_150/Block_123/Gemma_2'
P230817_step = r'F:/Chip/Experiment_2/Step_Gemma'

PFig = r'F:/Chip/Experiment_2/Analisis/Figures'

#Weight

P230817_Weight0 = r'F:/Growth_Cabinet/01082023/Weight_Experiment/Experiment_3/0_Weight_Gemma'
P230817_Weight2 = r'F:/Growth_Cabinet/01082023/Weight_Experiment/Experiment_3/2_Weight_Gemma'
P230817_Weight4 = r'F:/Growth_Cabinet/01082023/Weight_Experiment/Experiment_3/4_Weight_Gemma'

PFig = r'F:/Growth_Cabinet/01082023/Weight_Experiment/Experiment_3/Analisis/Figures'

#Wich Gemmae to save
#StackList230801_weight = ['G'+str(i) for i in range(1,32+1) if (i != 11)]
#StackList230801_ctrl = ['G'+str(i) for i in range(1,37+1)]

#Chip
StackList230817_0 = ['G'+str(i) for i in range(1,19+1) if (i != 9)] 
#StackList230801_ctrlV2 = ['G'+str(i) for i in range(1,52+1) if (i != 1 and i != 2 and i != 6 and i != 7 and i != 13 and i != 33 and i != 34 and i != 37 and i != 40 and i != 41 and i != 42 and i != 43 and i != 44 and i != 45 and i != 47 and i != 48 and i != 49 and i != 50 and i != 51)]
StackList230817_200 = ['G'+str(i) for i in range(1,36+1) if (i != 14 and i != 17 and i != 18 and i != 25 and i != 26 and i != 28)]
#StackList230817_150_V2 = ['G'+str(i) for i in range(1,62+1)]
StackList230817_step = ['G'+str(i) for i in range(1,37+1)  if (i != 1 and i != 2 and i != 23 and i != 24 and i != 36)]

#Weight
StackList230817_Weight0 = ['G'+str(i) for i in range(1,23+1)]
StackList230817_Weight2 = ['G'+str(i) for i in range(1,38+1) ]
StackList230817_Weight4 = ['G'+str(i) for i in range(1,45+1) if (i != 10)]



print('Data choice made.')