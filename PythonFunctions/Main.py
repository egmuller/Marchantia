# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:42:37 2022

@author: laplaud
"""

# Imports 
from GemmaeDetection import BinarizeStack, GetContours
from AreaCurveFitting import fitAreaGrowth,fitOsmoChoc, selectR2s, compareFit

import pandas as pd

#%% Wrapper functions for binarization and Area fit

# 1. Area growth

# stringName : name of experiment, StackList : list of videos to analyze, Path : path to data,
# Scale : in µm/px, FPH : frames per hour, Delay : delay between imbibtion and experiments start
# R2Threshold : threshold for data validation based on R2 of fit, 
# ToDo : indicator of which sub-code to run 'BCF' -> Binarization, Contours & Fits
# 'B' -> Binarization only, 'CF' -> contours & fits only, 'F' -> Fits only

# Kwargs : 'showHist' (True/false) for function selectR2s, 'debuglist' (list of integers) 
# to choose the images to analyse for debug plots 'HSVrange' (list of two 1x3 tuples)
# binarization min and max threshold values for the three channels of HSV image 
# 'fitwindow' (time in hours) to choose the window after dormancy exit on which to fit 

def BinarizeAndFitArea(stringName,StackList,Path,Scale,FPH,Delay,R2Threshold,ToDo, **kwargs):
    
      
    showHist = False
    DebugPlots = False
    HSVrange = [(25, 25, 70),(60, 120,220)]
    ImgList = [0, 20, 40]
    fitwindow=15
    
    for key, value in kwargs.items(): 
        if key == 'showHist':
            showHist = value
        elif key == 'debug':
            DebugPlots = value
        elif key == 'debuglist':
            ImgList = value 
        elif key == 'HSVrange':
            HSVrange = value
        elif key == 'fitwindow':
            fitwindow = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    if ToDo == 'BCF':
        DoBin = True
        DoCont = True
        DoFit = True
    elif ToDo == 'B':
        DoBin = True
        DoCont = False
        DoFit = False
    elif ToDo == 'CF':
        DoBin = False
        DoCont = True
        DoFit = True
    elif ToDo == 'F':
        DoBin = False
        DoCont = False
        DoFit = True
    else:
        raise NameError('ToDo variable is wrong')
    
    print('\n\n\n' + stringName + '\n\n')
    
    # Binarization of stacks
    if DoBin:
        BinarizeStack(StackList, Path, Scale,debug = DebugPlots, HSVrange = HSVrange, debuglist = ImgList)
    
    if DoCont:
                
        # Computing contours from binary
        CD,GD = GetContours(StackList,Path, Scale,FPH, debug=DebugPlots)

        # Saving all contours
        GD.to_csv(Path + '\\GlobalData' + stringName + '_AreaCont.csv',index_label = 'Ind')
        CD.to_csv(Path + '\\ContourData' + stringName + '_AreaCont.csv',index_label = 'Ind')

    if DoFit:
        
        GD = pd.read_csv(Path + '\\GlobalData' + stringName + '_AreaCont.csv', index_col = 'Ind')
        CD = pd.read_csv(Path + '\\ContourData' + stringName + '_AreaCont.csv',index_col = 'Ind')
        
        # Retrieve data on PPG position in chip        
        posinchip = pd.read_excel (Path + '\ChipPositions.xlsx', index_col='Name')        
        Rows = posinchip.loc[StackList].values[:,0]
        
        # Fitting area growth
        GD = fitAreaGrowth(StackList,Rows,GD,FPH,Delay, debug = DebugPlots,fitwindow = fitwindow)

        # Sorting based on fit quality
        GD, CD, R2s, goodList = selectR2s(GD,CD,R2Threshold,stringName, showHist = showHist)

        # Fit convergence 
        compareFit(GD, stringName)
        
        # Saving sorted contour and fit data
        GD.to_csv(Path + '\\GlobalData' + stringName + '_AreaFit.csv',index_label = 'Ind')
        CD.to_csv(Path + '\\ContourData' + stringName + '_AreaFit.csv',index_label = 'Ind')
    
    return

# 2. Osmotic chocs

# stringName : name of experiment, StackList : list of videos to analyze, Path : path to data,
# Scale : in µm/px, FPH : frames per hour, R2Threshold : threshold for data validation based on R2 of fit, 
# ToDo : indicator of which sub-code to run 'BCF' -> Binarization, Contours & Fits
# 'B' -> Binarization only, 'CF' -> contours & fits only, 'F' -> Fits only

# Kwargs : 'showHist' (True/false) for function selectR2s, 'debuglist' (list of integers) 
# to choose the images to analyse for debug plots 'HSVrange' (list of two 1x3 tuples)
# binarization min and max threshold values for the three channels of HSV image 
# 'FitIntervalComp/Rel (list of two ints) Images corresponding to times between 
# which to fit compression/relaxation

def BinarizeAndFitOsChoc(stringName,StackList,Path,Scale,FPH,R2Threshold,ToDo, **kwargs):
    
    
    showHist = False
    DebugPlots = False
    HSVrange = [(25, 25, 70),(60, 120,220)]
    ImgList = [0, 20, 40]
    FitIntervalComp = [2,25]
    FitIntervalRel = [29,55]
    
    for key, value in kwargs.items(): 
        if key == 'showHist':
            showHist = value
        elif key == 'debug':
            DebugPlots = value
        elif key == 'debuglist':
            ImgList = value 
        elif key == 'HSVrange':
            HSVrange = value
        elif key == 'FitIntervalComp':
            FitIntervalComp = value
        elif key == 'FitIntervalRel':
            FitIntervalRel = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    if ToDo == 'BCF':
        DoBin = True
        DoCont = True
        DoFit = True
    elif ToDo == 'B':
        DoBin = True
        DoCont = False
        DoFit = False
    elif ToDo == 'CF':
        DoBin = False
        DoCont = True
        DoFit = True
    elif ToDo == 'F':
        DoBin = False
        DoCont = False
        DoFit = True
    else:
        raise NameError('ToDo variable is wrong')
    
    print('\n\n\n' + stringName + '\n\n')
    
    # Binarization of stacks
    if DoBin:
        BinarizeStack(StackList, Path, Scale,debug = DebugPlots, HSVrange = HSVrange, debuglist = ImgList)
    
    if DoCont:
                
        # Computing contours from binary
        CD,GD = GetContours(StackList,Path, Scale,FPH, debug=DebugPlots)
        
        # Saving all contours
        GD.to_csv(Path + '\\GlobalData' + stringName + '_AreaCont.csv',index_label = 'Ind')
        CD.to_csv(Path + '\\ContourData' + stringName + '_AreaCont.csv',index_label = 'Ind')
        
        print('Contour saved')

    if DoFit:
        
        GD = pd.read_csv(Path + '\\GlobalData' + stringName + '_AreaCont.csv', index_col = 'Ind')
        CD = pd.read_csv(Path + '\\ContourData' + stringName + '_AreaCont.csv',index_col = 'Ind')
        
        GD = fitOsmoChoc(StackList,CD,GD,FPH,FitIntervalComp[0],FitIntervalComp[1],FitIntervalRel[0],FitIntervalRel[1],debug = DebugPlots)
        
        # Selecting only good R2s for both compand rel, could be changed in the future for more general plots
        GD, CD, R2s, goodList = selectR2s(GD, CD, R2Threshold, stringName,showHist=showHist)
        GD, CD, R2s, goodList = selectR2s(GD, CD, R2Threshold, stringName,showHist=showHist)
        
        # Saving sorted contour and fit data
        GD.to_csv(Path + '\\GlobalData' + stringName + '_AreaFit.csv',index_label = 'Ind')
        CD.to_csv(Path + '\\ContourData' + stringName + '_AreaFit.csv',index_label = 'Ind')
    
    return
