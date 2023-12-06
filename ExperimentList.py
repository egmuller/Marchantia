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



#Chip - Experiment 3
Delay_V2 = np.asarray([79+i*30 for i in range(0, 20)]+ [79+10*60+i*10 for i in range(0, 6)] + [79+10*60+60+i*30 for i in range(0,58)])
Delay_V3 = np.asarray([85+i*30 for i in range(0, 20)]+ [85+10*60+i*10 for i in range(0, 6)] + [85+10*60+60+i*30 for i in range(0,27)])

#Chip - Experiment 4/5
Delay_230908 = np.asarray([61+i*30 for i in range(0, 20)]+ [61+10*60+i*10 for i in range(0, 9)] + [61+10*60+90+i*30 for i in range(0,55)])


#Chip - Experiment 6
Delay_230925 = np.asarray([64+i*30 for i in range(0, 20)]+ [64+10*60+i*10 for i in range(0, 18)] + [64+10*60+180+i*30 for i in range(0,38)])


#Chip - Experiment 7
FPH171003 = 5
Delay_230928 = np.asarray([43+i*30 for i in range(0, 16)]+ [43+8*60+i*5 for i in range(0, 9)] + [43+8*60+60+i*30 for i in range(0,8)] + [43+8*60+60+240+i*10 for i in range(0,18)] + [43+8*60+60+240+180+i*30 for i in range(0,3)])



#Chip - Experiment 8
FPH171003 = 5
Delay_231003 = np.asarray([35+i*30 for i in range(0, 16)]+ [35+8*60+i*5 for i in range(0, 9)] + [35+8*60+60+i*30 for i in range(0,8)] + [35+8*60+60+240+i*5 for i in range(0,8)] + [35+8*60+60+240+40+i*10 for i in range(0,14)] + [35+8*60+60+240+40+140+i*30 for i in range(0,29)] )


#Chip - Experiment 9
FPH171003 = 5
Delay_231009 = np.asarray([36+i*30 for i in range(0, 16)]+ [36+8*60+i*1 for i in range(0, 45)] + [36+8*60+60+i*30 for i in range(0,8)] + [36+8*60+60+240+i*5 for i in range(0,8)] + [36+8*60+60+240+40+i*10 for i in range(0,14)] + [36+8*60+60+240+40+140+i*30 for i in range(0,38)] )


#Chip - Experiment 10
FPH171003 = 5
Delay_231016 = np.asarray([31+i*30 for i in range(0, 16)]+ [31+8*60+i*3 for i in range(0, 15)] + [31+8*60+60+i*30 for i in range(0,20)] + [31+8*60+60+9*60+50+i*5 for i in range(0,10)] + [31+8*60+60+9*60+50+50+i*10 for i in range(0,14)] + [31+8*60+60+9*60+50+50+140+i*30 for i in range(0,26)] )


#Chip - Experiment 11
FPH171003 = 5
Delay_231020 = np.asarray([27+i*30 for i in range(0, 16)]+ [27+8*60+i*3 for i in range(0, 15)] + [27+8*60+60+i*30 for i in range(0,20)] + [27+8*60+60+9*60+50+i*5 for i in range(0,10)] + [27+8*60+60+9*60+50+50+i*10 for i in range(0,14)] + [27+8*60+60+9*60+50+50+140+i*30 for i in range(0,26)] )


#Chip - Experiment 12
FPH171003 = 5
Delay_231024 = np.asarray([31+i*30 for i in range(0, 16)]+ [31+8*60+i*3 for i in range(0, 15)] + [31+8*60+60+i*10 for i in range(0,9)] + [31+8*60+60+90+i*5 for i in range(0,10)] + [31+8*60+60+90+50+i*10 for i in range(0,16)] + [31+8*60+60+90+50+160+i*30 for i in range(0,26)] )


#Chip - Experiment 14
FPH171003 = 5
Delay_231103 = np.asarray([39+i*30 for i in range(0, 20)]+ [39+10*60+i*10 for i in range(0, 6)] + [39+10*60+60+i*30 for i in range(0,39)])



#Chip - Experiment 15
FPH171003 = 5
Delay_231114 = np.asarray([54+i*30 for i in range(0, 8)]+ [54+4*60+i*10 for i in range(0, 6)] + [54+4*60+60+i*30 for i in range(0,54)])



#Chip - Experiment 16
FPH171003 = 5
Delay_231116 = np.asarray([37+i*30 for i in range(0, 12)]+ [37+6*60+i*10 for i in range(0, 6)] + [37+6*60+60+i*30 for i in range(0,54)])


#Chip - Experiment 17
FPH171003 = 5
Delay_231117 = np.asarray([31+i*30 for i in range(0, 16)]+ [31+8*60+i*3 for i in range(0, 15)] + [31+8*60+60+i*30 for i in range(0,20)] + [31+8*60+60+9*60+50+i*5 for i in range(0,10)] + [31+8*60+60+9*60+50+50+i*10 for i in range(0,14)] + [31+8*60+60+9*60+50+50+140+i*30 for i in range(0,26)] )



#Chip - Experiment 18
FPH171003 = 5
Delay_231127 = np.asarray([34+i*15 for i in range(0, 8)]+ [34+2*60+i*10 for i in range(0, 6)] + [34+2*60+60+i*30 for i in range(0,54)])


#Chip - Experiment 19
FPH171003 = 5
Delay_231205 = np.asarray([36+i*15 for i in range(0, 8)]+ [36+2*60+i*10 for i in range(0, 6)] + [36+2*60+60+i*30 for i in range(0,54)])




#Growth Cabinet

FPHCabinet = 6 # Frames per hour
Delay_Cabinet = 0 # Delay between imbibition and experiment start, in min


#Growth Cabinet - Experiment 1

Scale_Weight230817_0 = 1.15 # Spatial scale (µm/px)
Scale_Weight230817_4 = 1.13 # Spatial scale (µm/px) 

#Growth Cabinet - Experiment 2

Scale_Weight230823_0 = 1.14 # Spatial scale (µm/px)
Scale_Weight230823_2 = 1.12 # Spatial scale (µm/px)
Scale_Weight230823_4 = 1.18 # Spatial scale (µm/px)


#Growth Cabinet - Experiment 3


Scale_Weight230826_0 = 0.92 # Spatial scale (µm/px)
Scale_Weight230826_2 = 0.90 # Spatial scale (µm/px) 
Scale_Weight230826_4 = 0.92 # Spatial scale (µm/px)



#Growth Cabinet - Experiment 4

Scale_Weight230823_0 = 0.98 # Spatial scale (µm/px)
Scale_Weight230823_2 = 0.98 # Spatial scale (µm/px) 
Scale_Weight230823_4 = 1.03 # Spatial scale (µm/px)



#Growth Cabinet - Experiment 5

Scale_Weight230830_0 = 1.05 # Spatial scale (µm/px)
Scale_Weight230830_2 = 0.96 # Spatial scale (µm/px) 
Scale_Weight230830_4 = 0.95 # Spatial scale (µm/px)



#Growth Cabinet - Experiment 6

Scale_Weight230906_0 = 0.92 # Spatial scale (µm/px)
Scale_Weight230906_2 = 0.93 # Spatial scale (µm/px) 
Scale_Weight230906_4 = 0.90 # Spatial scale (µm/px)


#Growth Cabinet - Experiment 7

Scale_Weight230907_0 = 0.98 # Spatial scale (µm/px)
Scale_Weight230907_4R = 1.07 # Spatial scale (µm/px) 
Scale_Weight230907_4 = 1.04 # Spatial scale (µm/px)

#Growth Cabinet - Experiment 8

Scale_Weight230913_0 = 1.04 # Spatial scale (µm/px)
Scale_Weight230913_4R = 0.97 # Spatial scale (µm/px) 
Scale_Weight230913_4 = 1.05 # Spatial scale (µm/px)


#Growth Cabinet - Experiment 9

Scale_Weight230918_0 = 1.07 # Spatial scale (µm/px)
Scale_Weight230918_4R = 1.06 # Spatial scale (µm/px) 
Scale_Weight230918_4 = 1.05 # Spatial scale (µm/px)

#Growth Cabinet - Experiment 10

Scale_Weight230922_0 = 1.09 # Spatial scale (µm/px)
Scale_Weight230922_4R = 1.04 # Spatial scale (µm/px) 
Scale_Weight230922_4 = 1.06 # Spatial scale (µm/px)


#Growth Cabinet - Experiment 11

Scale_Weight231002_0 = 1.13 # Spatial scale (µm/px)
Scale_Weight231002_4R = 1.04 # Spatial scale (µm/px) 
Scale_Weight231002_4 = 1.01 # Spatial scale (µm/px)


#Growth Cabinet - Experiment 12

Scale_Weight231004_0 = 0.98 # Spatial scale (µm/px)
Scale_Weight231004_4R = 0.93 # Spatial scale (µm/px) 
Scale_Weight231004_4 = 0.92 # Spatial scale (µm/px)


#Growth Cabinet - Experiment 13

Scale_Weight231010_0 = 0.97 # Spatial scale (µm/px)
Scale_Weight231010_4R = 0.96 # Spatial scale (µm/px) 
Scale_Weight231010_4 = 0.94 # Spatial scale (µm/px)



#Chip - Experiment 1

P230817_0 = r'F:/Chip/Experiment_1/Chamber_0/Block_123/Gemma'
P230817_150 = r'F:/Chip/Experiment_1/Chamber_150/Block_123/Gemma'
#P230817_150_V2 = r'F:/Chip/Chamber_150/Block_123/Gemma_2'
P230817_step = r'F:/Chip/Experiment_1/Chamber_Jump/Block_123/Gemma'

PFig1C = r'F:/Chip/Experiment_1/Analisis/Figures'


#Chip - Experiment 2

P230829_0 = r'F:/Chip/Experiment_2/0M_Gemma'
P230829_200 = r'F:/Chip/Experiment_2/200M_Gemma'
P230829_step = r'F:/Chip/Experiment_2/Step_Gemma'

PFig2C = r'F:/Chip/Experiment_2/Analisis/Figures'

#Chip - Experiment 3

P230904_0 = r'F:/Chip/Experiment_3/0M_Gemma'
P230904_200 = r'F:/Chip/Experiment_3/200M_Gemma'
P230904_step = r'F:/Chip/Experiment_3/Step_Gemma'

PFig3C = r'F:/Chip/Experiment_3/Analisis/Figures'

#Chip - Experiment 4

P230908_0 = r'F:/Chip/Experiment_4/0M_Gemma'
P230908_400 = r'F:/Chip/Experiment_4/400M_Gemma'
P230908_step = r'F:/Chip/Experiment_4/Step_Gemma'

PFig4C = r'F:/Chip/Experiment_4/Analisis/Figures'

#Chip - Experiment 5

P230914_0 = r'F:/Chip/Experiment_5/0M_Gemma'
P230914_400 = r'F:/Chip/Experiment_5/400M_Gemma'
P230914_step = r'F:/Chip/Experiment_5/Step_Gemma'

PFig5C = r'F:/Chip/Experiment_5/Analisis/Figures'

#Chip - Experiment 6

P230925_0 = r'F:/Chip/Experiment_6/0M_Gemma'
P230925_100 = r'F:/Chip/Experiment_6/100M_Gemma'
P230925_step = r'F:/Chip/Experiment_6/Step_Gemma'

PFig6C = r'F:/Chip/Experiment_6/Analisis/Figures'

#Chip - Experiment 7

P230928_0 = r'F:/Chip/Experiment_7/0M_Gemma'
P230928_step = r'F:/Chip/Experiment_7/Step_Gemma'
P230928_step_shock = r'F:/Chip/Experiment_7/Step_Gemma_Shock'

PFig7C = r'F:/Chip/Experiment_7/Analisis/Figures'

#Chip - Experiment 8

P231003_0 = r'F:/Chip/Experiment_8/0M_Gemma'
P231003_step = r'F:/Chip/Experiment_8/Step_Gemma'
P231003_step_shock = r'F:/Chip/Experiment_8/Step_Gemma_Shock'

PFig8C = r'F:/Chip/Experiment_8/Analisis/Figures'

#Chip - Experiment 9

P231009_0 = r'F:/Chip/Experiment_9/0M_Gemma'
P231009_step = r'F:/Chip/Experiment_9/Step_Gemma'
P231009_step_shock = r'F:/Chip/Experiment_9/Step_Gemma_Shock'

PFig9C = r'F:/Chip/Experiment_9/Analisis/Figures'


#Chip - Experiment 10

P231016_0 = r'F:/Chip/Experiment_10/0M_Gemma'
P231016_step = r'F:/Chip/Experiment_10/Step_Gemma'
P231016_step_shock = r'F:/Chip/Experiment_10/Step_Gemma_Shock'

PFig10C = r'F:/Chip/Experiment_10/Analisis/Figures'


#Chip - Experiment 11

P231020_0 = r'F:/Chip/Experiment_11/0M_Gemma'
P231020_step = r'F:/Chip/Experiment_11/Step_Gemma'
P231020_step_shock = r'F:/Chip/Experiment_11/Step_Gemma_Shock'

PFig11C = r'F:/Chip/Experiment_11/Analisis/Figures'



#Chip - Experiment 12

P231024_0 = r'F:/Chip/Experiment_12/0M_Gemma'
P231024_step = r'F:/Chip/Experiment_12/Step_Gemma'
P231024_step_shock = r'F:/Chip/Experiment_12/Step_Gemma_Shock'

PFig12C = r'F:/Chip/Experiment_12/Analisis/Figures'


#Chip - Experiment 14

P231103_0 = r'F:/Chip/Experiment_14/0M_Gemma'
P231103_200 = r'F:/Chip/Experiment_14/200M_Gemma'
P231103_step = r'F:/Chip/Experiment_14/Step_Gemma'

PFig14C = r'F:/Chip/Experiment_14/Analisis/Figures'


#Chip - Experiment 15

P231114_0 = r'F:/Chip/Experiment_15/0M_Gemma'
P231114_400 = r'F:/Chip/Experiment_15/400M_Gemma'
P231114_step = r'F:/Chip/Experiment_15/Step_Gemma'


PFig15C = r'F:/Chip/Experiment_15/Analisis/Figures'


#Chip - Experiment 16

P231116_0 = r'F:/Chip/Experiment_16/0M_Gemma'
P231116_400 = r'F:/Chip/Experiment_16/400M_Gemma'
P231116_step = r'F:/Chip/Experiment_16/Step_Gemma'


PFig16C = r'F:/Chip/Experiment_16/Analisis/Figures'

#Chip - Experiment 17

P231117_0 = r'F:/Chip/Experiment_17/0M_Gemma'
P231117_step = r'F:/Chip/Experiment_17/Step_Gemma'
P231117_step_shock = r'F:/Chip/Experiment_17/Step_Gemma_Shock'

PFig17C = r'F:/Chip/Experiment_17/Analisis/Figures'

#Chip - Experiment 18

P231127_0 = r'F:/Chip/Experiment_18/0M_Gemma'
P231127_400 = r'F:/Chip/Experiment_18/400M_Gemma'
P231127_step = r'F:/Chip/Experiment_18/Step_Gemma'

PFig18C = r'F:/Chip/Experiment_18/Analisis/Figures'


#Chip - Experiment 19

P231205_0 = r'F:/Chip/Experiment_19/0M_Gemma'
P231205_400 = r'F:/Chip/Experiment_19/400M_Gemma'
P231205_step = r'F:/Chip/Experiment_19/Step_Gemma'

PFig19C = r'F:/Chip/Experiment_19/Analisis/Figures'




#Growth Cabinet - Experiment 1

P230817_Weight0 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_1/Gemma/Gemma_No_Weight'
P230817_Weight4 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_1/Gemma/Gemma_Weight'

PFig1W = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_1/Analisis/Figures'



#Growth Cabinet - Experiment 2

P230823_Weight0 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_2/Gemma/Gemma_0Weight'
P230823_Weight2 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_2/Gemma/Gemma_2Weight'
P230823_Weight4 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_2/Gemma/Gemma_4Weight'

PFig2W = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_2/Analisis/Figures'

#Growth Cabinet - Experiment 3

P230826_Weight0 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_3/0_Weight_Gemma'
P230826_Weight2 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_3/2_Weight_Gemma'
P230826_Weight4 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_3/4_Weight_Gemma'

PFig3W = r'F:/Growth_CabinetWeight_Experiment/Experiment_3/Analisis/Figures'


#Growth Cabinet - Experiment 4

P230829_Weight0 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_4/0W_Gemma'
P230829_Weight2 = r'F:/Growth_CabinetWeight_Experiment/Experiment_4/2W_Gemma'
P230829_Weight4 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_4/4W_Gemma'

PFig4W = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_4/Analisis/Figures'



#Growth Cabinet - Experiment 5

P230830_Weight0 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_5/0W_Gemma'
P230830_Weight2 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_5/2W_Gemma'
P230830_Weight4 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_5/4W_Gemma'

PFig5W = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_5/Analisis/Figures'


#Growth Cabinet - Experiment 6

P230906_Weight0 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_6/0W_Gemma'
P230906_Weight2 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_6/2W_Gemma'
P230906_Weight4 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_6/4W_Gemma'

PFig6W = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_6/Analisis/Figures'



#Growth Cabinet - Experiment 7

P230907_Weight0 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_7/0W_Gemma'
P230907_Weight4R = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_7/4WR_Gemma'
P230907_Weight4 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_7/4W_Gemma'

PFig7W = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_7/Analisis/Figures'


#Growth Cabinet - Experiment 8


P230913_Weight0 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_8/0W_Gemma'
P230913_Weight4R = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_8/4WR_Gemma'
P230913_Weight4 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_8/4W_Gemma'

PFig8W = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_8/Analisis/Figures'



#Growth Cabinet - Experiment 9


P230918_Weight0 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_9/0W_Gemma'
P230918_Weight4R = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_9/4WR_Gemma'
P230918_Weight4 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_9/4W_Gemma'

PFig9W = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_9/Analisis/Figures'



#Growth Cabinet - Experiment 10


P230922_Weight0 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_10/0W_Gemma'
P230922_Weight4R = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_10/4WR_Gemma'
P230922_Weight4 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_10/4W_Gemma'

PFig10W = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_10/Analisis/Figures'



#Growth Cabinet - Experiment 11


P231002_Weight0 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_11/0W_Gemma'
P231002_Weight4R = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_11/4WR_Gemma'
P231002_Weight4 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_11/4W_Gemma'

PFig11W = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_11/Analisis/Figures'


#Growth Cabinet - Experiment 12


P231004_Weight0 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_12/0W_Gemma'
P231004_Weight4R = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_12/4WR_Gemma'
P231004_Weight4 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_12/4W_Gemma'

PFig12W = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_12/Analisis/Figures'

#Growth Cabinet - Experiment 13

P231010_Weight0 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_13/0W_Gemma'
P231010_Weight4R = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_13/4WR_Gemma'
P231010_Weight4 = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_13/4W_Gemma'

PFig13W = r'F:/Growth_Cabinet/Weight_Experiment/Experiment_12/Analisis/Figures'



#Chip - Experiment 1

StackList230817_0 = ['G'+str(i) for i in range(1,33+1) ] 
StackList230817_150 = ['G'+str(i) for i in range(1,9+1) ]
#StackList230817_150_V2 = ['G'+str(i) for i in range(1,62+1)]
StackList230817_step = ['G'+str(i) for i in range(1,44+1) ]

#Chip - Experiment 2

StackList230829_0 = ['G'+str(i) for i in range(1,19+1) if (i != 9)] 
StackList230829_200 = ['G'+str(i) for i in range(1,36+1) if (i != 14 and i != 17 and i != 18 and i != 25 and i != 26 and i != 28)]
StackList230829_step = ['G'+str(i) for i in range(1,37+1) if (i != 1 and i != 2 and i != 23 and i != 24 and i != 36)]

#Chip - Experiment 3

StackList230904_0 = ['G'+str(i) for i in range(1,31+1) if (i != 1 and i != 4  and i != 7 and i != 10 and i != 16 and i != 18 and i != 27 and i != 28 and i != 29) ] 
StackList230904_200 = ['G'+str(i) for i in range(1,33+1) if (i != 4 and i != 5 and i != 11 and i!= 15 and i != 22 and i != 24)]
StackList230904_step = ['G'+str(i) for i in range(1,41+1) if (i != 7 and i != 12 and i != 18 and i != 23 and i != 24 and i != 26 and i != 27 and i != 28 and i != 29 and i != 31 and i != 32 and i != 34 and i != 35 and i != 36 and i != 37  and i != 38 and i != 39)]

#Chip - Experiment 4

StackList230908_0 = ['G'+str(i) for i in range(1,44+1) if (i != 18 and i != 41) ] 
StackList230908_400 = ['G'+str(i) for i in range(1,33+1) if (i != 10 and i != 17 and i != 25 and i != 26) ]
StackList230908_step = ['G'+str(i) for i in range(1,31+1) if (i != 1 and i != 19 and i != 20 and i != 21 and i != 25 and i != 31) ]

#Chip - Experiment 5

StackList230914_0 = ['G'+str(i) for i in range(1,37+1) if (i != 2 and i != 4 and i != 17 and i != 25 and i != 27 and i != 29 and i != 32) ] 
StackList230914_400 = ['G'+str(i) for i in range(1,33+1) if (i != 2 and i != 6 and i != 8 and i != 23) ]
StackList230914_step = ['G'+str(i) for i in range(1,10+1)]

#Chip - Experiment 6

StackList230925_0 = ['G'+str(i) for i in range(1,34+1) if (i != 18 and i != 19) ] 
StackList230925_100 = ['G'+str(i) for i in range(1,32+1) if (i != 4 and i != 6 and i != 8 and i != 17 and i != 18 and i != 20)]
StackList230925_step = ['G'+str(i) for i in range(1,37+1) if (i != 14 and i != 15)]

#Chip - Experiment 7

StackList230928_0 = ['G'+str(i) for i in range(1,34+1) if (i != 16 and i != 17) ] 
StackList230928_step = ['G'+str(i) for i in range(1,50+1) if (i != 4 and i != 8 and i != 12 and i != 13 and i != 30 and i != 37) ]
StackList230928_step_shock = ['G'+str(i) for i in range(1,50+1) if (i != 4 and i != 8 and i != 12 and i != 13 and i != 30 and i != 37) ]

#Chip - Experiment 8

StackList231003_0 = ['G'+str(i) for i in range(1,48+1) if (i != 1 and i != 39 and i != 46 ) ] 
StackList231003_step = ['G'+str(i) for i in range(1,51+1) if (i != 6 and i != 7 and i != 14  and i != 15 and i != 35 and i != 43 ) ]
StackList231003_step_shock = ['G'+str(i) for i in range(1,51+1) if (i != 6 and i != 7 and i != 14  and i != 15 and i != 35 and i != 43 )  ]

#Chip - Experiment 9


StackList231009_0 = ['G'+str(i) for i in range(1,45+1) if (i != 7 and i != 29 and i != 30 and i != 39 and i != 43 and i != 45) ]
StackList231009_step = ['G'+str(i) for i in range(1,34+1) if (i != 14 and i != 17)  ] 
StackList231009_step_shock = ['G'+str(i) for i in range(1,34+1) if (i != 14 and i != 17) ]

#Chip - Experiment 10

StackList231016_0 = ['G'+str(i) for i in range(1,37+1)  if (i != 9 and i != 12 and i != 19  ) ]
StackList231016_step = ['G'+str(i) for i in range(1,38+1)  if (i != 4 and i != 10 and i != 34 and i != 36 and i != 38)  ] 
StackList231016_step_shock = ['G'+str(i) for i in range(1,38+1)  if (i != 4 and i != 10 and i != 34 and i != 36 and i != 38)]

#Chip - Experiment 11

StackList231020_0 = ['G'+str(i) for i in range(1,32+1)]
StackList231020_step = ['G'+str(i) for i in range(1,42+1)  ] 
StackList231020_step_shock = ['G'+str(i) for i in range(1,42+1)]

#Chip - Experiment 12

StackList231024_0 = ['G'+str(i) for i in range(1,36+1) if (i != 15 and i != 27 and i != 32) ]
StackList231024_step = ['G'+str(i) for i in range(1,36+1) if (i !=1  and i !=4  and i !=11 and i !=33)] 
StackList231024_step_shock = ['G'+str(i) for i in range(1,36+1) if (i !=1  and i !=4  and i !=11 and i !=33)]


#Chip - Experiment 14

StackList231103_0 = ['G'+str(i) for i in range(1,24+1)  if (i != 15) ]
StackList231103_200 = ['G'+str(i) for i in range(1,16+1) ] 
StackList231103_step = ['G'+str(i) for i in range(1,20+1) if (i != 2 and i != 19)]

#Chip - Experiment 15

StackList231114_0 = ['G'+str(i) for i in range(1,34+1)  ]
StackList231114_400 = ['G'+str(i) for i in range(1,21+1) if (i != 5 and  i != 10 and i != 15) ] 
StackList231114_step = ['G'+str(i) for i in range(1,21+1) if (i != 14 and i != 20) ]


#Chip - Experiment 16

StackList231116_0 = ['G'+str(i) for i in range(1,49+1) if (i != 6 and  i != 20 and  i != 21 and  i != 25 and  i != 27) ]
StackList231116_400 = ['G'+str(i) for i in range(1,41+1) if (i != 7 and i != 22) ] 
StackList231116_step = ['G'+str(i) for i in range(1,27+1) if (i != 8 and  i != 16 and  i != 24) ]


#Chip - Experiment 17

StackList231117_0 = ['G'+str(i) for i in range(1,41+1) if (i != 36 )  ]
StackList231117_step = ['G'+str(i) for i in range(1,40+1) if (i != 5 and  i != 11 and  i != 13 )  ]
StackList231117_step_shock = ['G'+str(i) for i in range(1,40+1) if (i != 5 and  i != 11 and  i != 13 ) ]


#Chip - Experiment 18

StackList231127_0 = ['G'+str(i) for i in range(1,31+1) if (i != 24 and  i != 27 ) ]
StackList231127_400 = ['G'+str(i) for i in range(1,53+1) if (i != 2 and  i != 36 and  i != 46 ) ] 
StackList231127_step = ['G'+str(i) for i in range(1,29+1) if ( i != 20) ]



#Chip - Experiment 19

StackList231205_0 = ['G'+str(i) for i in range(1,21+1) if (i != 4 and  i != 13) ]
StackList231205_400 = ['G'+str(i) for i in range(1,36+1) if (i !=6 and  i != 12)] 
StackList231205_step = ['G'+str(i) for i in range(1,27+1) if (i !=6 and  i != 20)]







#Growth Cabinet - Experiment 1
StackList230817_Weight0 = ['G'+str(i) for i in range(1,52+1)]
StackList230817_Weight4 = ['G'+str(i) for i in range(1,73+1)]


#Growth Cabinet - Experiment 2
StackList230823_Weight0 = ['G'+str(i) for i in range(1,35+1) ]
StackList230823_Weight2 = ['G'+str(i) for i in range(1,45+1) ]
StackList230823_Weight4 = ['G'+str(i) for i in range(1,55+1) ]


#Growth Cabinet - Experiment 3
StackList230826_Weight0 = ['G'+str(i) for i in range(1,23+1) ]
StackList230826_Weight2 = ['G'+str(i) for i in range(1,38+1) ]
StackList230826_Weight4 = ['G'+str(i) for i in range(1,45+1) if (i != 10)]


#Growth Cabinet - Experiment 4
StackList230829_Weight0 = ['G'+str(i) for i in range(1,33+1)if (i != 21)]
StackList230829_Weight2 = ['G'+str(i) for i in range(1,33+1)if (i != 29)]
StackList230829_Weight4 = ['G'+str(i) for i in range(1,40+1)]


#Growth Cabinet - Experiment 5
StackList230830_Weight0 = ['G'+str(i) for i in range(1,42+1)]
StackList230830_Weight2 = ['G'+str(i) for i in range(1,33+1) if (i != 6 and i != 33)]
StackList230830_Weight4 = ['G'+str(i) for i in range(1,43+1)if (i != 21 )]


#Growth Cabinet - Experiment 6
StackList230906_Weight0 = ['G'+str(i) for i in range(1,33+1)if (i != 20 )]
StackList230906_Weight2 = ['G'+str(i) for i in range(1,42+1)]
StackList230906_Weight4 = ['G'+str(i) for i in range(1,33+1)]


#Growth Cabinet - Experiment 7
StackList230907_Weight0 = ['G'+str(i) for i in range(1,38+1) if (i != 30  and i != 33 and i != 35 and i != 37)]
StackList230907_Weight4R = ['G'+str(i) for i in range(1,42+1) if (i != 13) ]
StackList230907_Weight4 = ['G'+str(i) for i in range(1,49+1) if (i != 2  and i != 15  and i != 24  and i != 30  and i != 43  and i != 46 )]

#Growth Cabinet - Experiment 8
StackList230913_Weight0 = ['G'+str(i) for i in range(1,50+1) if (i != 17 and i != 35 and i != 37)]
StackList230913_Weight4R = ['G'+str(i) for i in range(1,39+1) if (i != 10 and i != 20 and i != 28  and i != 31 )]
StackList230913_Weight4 = ['G'+str(i) for i in range(1,48+1) if (i != 8 and i != 27 and i != 31 and i != 33 and i != 39 and i != 42 and i != 44)]

#Growth Cabinet - Experiment 9
StackList230918_Weight0 = ['G'+str(i) for i in range(1,36+1) ]
StackList230918_Weight4R = ['G'+str(i) for i in range(1,38+1) if (i != 33) ]
StackList230918_Weight4 = ['G'+str(i) for i in range(1,36+1) ]

#Growth Cabinet - Experiment 10
StackList230922_Weight0 = ['G'+str(i) for i in range(1,49+1) if (i != 11 and i != 13 and i != 17 and i != 19 and i != 21 and i != 32 and i != 38)  ]
StackList230922_Weight4R = ['G'+str(i) for i in range(1,40+1) if (i != 6 and i != 10 and i != 17 and i != 18) ]
StackList230922_Weight4 = ['G'+str(i) for i in range(1,51+1)  if (i != 18 and i != 23 and i != 25 and i != 27 and i != 50 and i != 51) ]

#Growth Cabinet - Experiment 11
StackList231002_Weight0 = ['G'+str(i) for i in range(1,26+1) if (i != 2 and i != 23)]
StackList231002_Weight4R = ['G'+str(i) for i in range(1,29+1) if (i != 22 and i != 27)  ]
StackList231002_Weight4 = ['G'+str(i) for i in range(1,32+1) if (i != 19 and i != 21 and i != 27 and i != 28 and  i != 32) ]


#Growth Cabinet - Experiment 11
StackList231004_Weight0 = ['G'+str(i) for i in range(1,28+1) if (i != 3  and i != 8  and i != 10  and i != 12  and i != 13  and i != 15) ]
StackList231004_Weight4R = ['G'+str(i) for i in range(1,32+1) if (i != 5  and i != 8 and i != 11 and i != 14 )]
StackList231004_Weight4 = ['G'+str(i) for i in range(1,21+1) if (i != 2 and i != 13) ]


#Growth Cabinet - Experiment 12
StackList231010_Weight0 = ['G'+str(i) for i in range(1,24+1) ]
StackList231010_Weight4R = ['G'+str(i) for i in range(1,40+1) if (i != 19 and i != 27 ) ]
StackList231010_Weight4 = ['G'+str(i) for i in range(1,24+1) ]

print('Data choice made.')