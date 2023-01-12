# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:42:37 2022

@author: laplaud
"""

# Imports 

from GemmaeDetection import BinarizeStack, GetContours, FindChipPos
from AreaCurveFitting import fitAreaGrowth,fitOsmoChoc,selectR2s
from ContourAnalysis import getLandmarks, rotateAndCenterShape, curvAbsci
from ContourQuantifs import computeMeanContourTime, DistToMean, computeSym, GetGrowth

import numpy as np
import pandas as pd

import os



from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#%% Wrapper functions for binarization and Area fit

# 1. Area growth

# stringName : name of experiment, StackList : list of videos to analyze, Path : path to data,
# Scale : in µm/px, FPH : frames per hour, Delay : delay between imbibtion and experiments start
# R2Threshold : threshold for data validation based on R2 of fit, 
# Ori : Orientation of the chip Horizontal chip : H + direction of traps ('H>', 'H<', 'V^', 'Vv')
# ToDo : indicator of which sub-code to run 'BCF' -> Binarization, Contours & Fits
# 'B' -> Binarization only, 'CF' -> contours & fits only, 'F' -> Fits only

# Kwargs : 'showHist' (True/false) for function selectR2s, 'debuglist' (list of integers) 
# to choose the images to analyse for debug plots 'HSVrange' (list of two 1x3 tuples)
# binarization min and max threshold values for the three channels of HSV image 
# 'fitwindow' (time in hours) to choose the window after dormancy exit on which to fit 

def BinarizeAndFitArea(stringName,StackList,Path,Scale,FPH,Delay,R2Threshold,Ori,ToDo, **kwargs):
    
      
    DebugAll = False
    DebugPlots = False
    ValidPlots = False
    HSVrange = [(25, 25, 70),(60, 120,220)]
    ImgList = [0, 20, 40]
    fitwindow=15
    
    for key, value in kwargs.items(): 
        if key == 'debugAll':
            DebugAll = value
        elif key == 'debug':
            DebugPlots = value
        elif key == 'plotValid':
            ValidPlots = value
        elif key == 'saveWB':
            saveWB = value
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
    elif ToDo == 'BC':
        DoBin = True
        DoCont = True
        DoFit = False
    elif ToDo == 'C':
        DoBin = False
        DoCont = True
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
    
    print('\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')
    print('\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('\n\nAnalyzing experiment : ' + stringName + '\n')
    
    # Binarization of stacks
    if DoBin:
        BinarizeStack(StackList, Path, Scale,debug = DebugAll, HSVrange = HSVrange, debuglist = ImgList)
        print('\n\n')
    
    if DoCont:
                
        # Computing contours from binary
        CD,GD = GetContours(StackList,Path, Scale,FPH, debug=DebugAll)

        # Saving all contours
        GD.to_csv(Path + '\\GlobalData' + stringName + '_AreaCont.csv',index_label = 'Ind')
        CD.to_csv(Path + '\\ContourData' + stringName + '_AreaCont.csv',index_label = 'Ind')
        print('\n\n')

    if DoFit:
        
        GD = pd.read_csv(Path + '\\GlobalData' + stringName + '_AreaCont.csv', index_col = 'Ind')
        CD = pd.read_csv(Path + '\\ContourData' + stringName + '_AreaCont.csv',index_col = 'Ind')
        print('\n\n')
        
        # Retrieve data on PPG position in chip 
        if os.path.exists(Path + '\ChipPositions.xlsx'):
            posinchip = pd.read_excel (Path + '\ChipPositions.xlsx', index_col='Name') 
            Rows = posinchip.loc[StackList].values[:,0]
        else:
            print('Finding PPGs position in chip...',end='')
            Rows = FindChipPos(StackList,Path,Ori)
            print('Done\n')
        print('\n\n')
        
        # Fitting area growth
        GD = fitAreaGrowth(StackList,Rows,GD,FPH,Delay,R2Threshold,ValidPlots= ValidPlots, debugall = DebugAll, debug = DebugPlots,fitwindow = fitwindow)
        
        GD.loc[:,'Expe'] = stringName
        
        print('\n\n')

        # Sorting based on fit quality
        GD, CD, R2s, goodList = selectR2s(GD,CD,R2Threshold,stringName, showHist = DebugPlots)
        print('\n\n')
        
        # Saving sorted contour and fit data
        GD.to_csv(Path + '\\GlobalData' + stringName + '_AreaFit.csv',index_label = 'Ind')
        CD.to_csv(Path + '\\ContourData' + stringName + '_AreaFit.csv',index_label = 'Ind')
        print('\n\n')
    
    return

# 2. Osmotic chocs

# stringName : name of experiment, StackList : list of videos to analyze, Path : path to data,
# Scale : in µm/px, FPH : frames per hour, R2Threshold : threshold for data validation based on R2 of fit, 
# Ori : Orientation of the chip Horizontal chip : H + direction of traps ('H>', 'H<', 'V^', 'Vv')
# ToDo : indicator of which sub-code to run 'BCF' -> Binarization, Contours & Fits
# 'B' -> Binarization only, 'CF' -> contours & fits only, 'F' -> Fits only

# Kwargs : 'showHist' (True/false) for function selectR2s, 'debuglist' (list of integers) 
# to choose the images to analyse for debug plots 'HSVrange' (list of two 1x3 tuples)
# binarization min and max threshold values for the three channels of HSV image 
# 'FitIntervalComp/Rel (list of two ints) Images corresponding to times between 
# which to fit compression/relaxation

def BinarizeAndFitOsChoc(stringName,StackList,Path,Scale,FPH,R2Threshold,Ori,ToDo, **kwargs):
    
    
    showHist = False
    DebugPlots = False
    HSVrange = [(25, 25, 70),(60, 120,220)]
    ImgList = [0, 20, 40]
    FitIntervalComp = [0,25]
    FitIntervalRel = [25,55]
    TstartComp = 3
    TstartRel = 9
    RelValidation = True
    saveWB = False
    IgnoredCompPoints = []
    IgnoredRelPoints = []
    Concentration = 100
    
    for key, value in kwargs.items(): 
        if key == 'showHist':
            showHist = value
        elif key == 'debug':
            DebugPlots = value
        elif key == 'debuglist':
            ImgList = value 
        elif key == 'saveWB':
            saveWB = value
        elif key == 'HSVrange':
            HSVrange = value
        elif key == 'FitIntervalComp':
            FitIntervalComp = value
        elif key == 'FitIntervalRel':
            FitIntervalRel = value
        elif key == 'TstartComp':
            TstartComp = value
        elif key == 'TstartRel':
            TstartRel = value
        elif key == 'RelValidation':
            RelValidation = value
        elif key == 'ignoreCompTimePoints':
            IgnoredCompPoints = value
        elif key == 'ignoreRelTimePoints':
            IgnoredRelPoints = value
        elif key == "C_osmo":
            Concentration = value
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
    elif ToDo == 'BC':
        DoBin = True
        DoCont = True
        DoFit = False
    elif ToDo == 'C':
        DoBin = False
        DoCont = True
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
        BinarizeStack(StackList, Path, Scale,debug = DebugPlots, HSVrange = HSVrange, debuglist = ImgList,saveWB = saveWB)
    
    if DoCont:
                
        # Computing contours from binary
        CD,GD = GetContours(StackList,Path, Scale,FPH, debug=DebugPlots)
        
        # Saving all contours
        GD.to_csv(Path + '\\GlobalData' + stringName + '_AreaCont.csv',index_label = 'Ind')
        CD.to_csv(Path + '\\ContourData' + stringName + '_AreaCont.csv',index_label = 'Ind')
        
        print('Contour saved')

    if DoFit:
        
        # Retrieve data on PPG position in chip 
        if os.path.exists(Path + '\ChipPositions.xlsx'):
            posinchip = pd.read_excel (Path + '\ChipPositions.xlsx', index_col='Name') 
            Rows = posinchip.loc[StackList].values[:,0]
        else:
            print('Finding PPGs position in chip...',end='')
            Rows = FindChipPos(StackList,Path,Ori)
            print('Done\n')
        print('\n\n')
        
        GD = pd.read_csv(Path + '\\GlobalData' + stringName + '_AreaCont.csv', index_col = 'Ind')
        CD = pd.read_csv(Path + '\\ContourData' + stringName + '_AreaCont.csv',index_col = 'Ind')
        
        GD = fitOsmoChoc(StackList,Rows,CD,GD,FPH,FitIntervalComp[0],FitIntervalComp[1],
                         TstartComp,FitIntervalRel[0],FitIntervalRel[1],TstartRel,debug = DebugPlots,
                         ignoredCTP = IgnoredCompPoints, ignoredRTP = IgnoredRelPoints, C_Osmo = Concentration)
        
        GD.loc[:,'Expe'] = stringName
        
        # Selecting only good R2s for both compand rel, could be changed in the future for more general plots
        GD, CD, R2s, goodList = selectR2s(GD, CD, R2Threshold, stringName,showHist=showHist)
        if RelValidation:
            GD, CD, R2s, goodList = selectR2s(GD, CD, R2Threshold, stringName,showHist=showHist,key = 'fitR2rel')
        
        # Saving sorted contour and fit data
        GD.to_csv(Path + '\\GlobalData' + stringName + '_AreaFit.csv',index_label = 'Ind')
        CD.to_csv(Path + '\\ContourData' + stringName + '_AreaFit.csv',index_label = 'Ind')
    
    return


 

#%% Wrapper functions for analysis of contours


def ParametriseContour(stringName,Path,dateCond,Scale,Todo, **kwargs):
    
    doL   = False
    doLPR = False
    doPR  = False
    doR   = False
    
    if Todo == 'LPR':        
        doLPR = True
    elif Todo == 'PR':
        doPR  = True
    elif Todo == 'R':
        doR   = True
    elif Todo == 'L':
        doL   = True
        
    DebugPlots = False  
    LdmkPlots = True
    Dmax = 20   
    Dmax2 = 20   
    AUTO = False
    OverwriteData = False
    FirstSlice = False
    
    for key, value in kwargs.items(): 
        if key == 'debug':
            DebugPlots = value
        elif key == 'ldmkplots':
            LdmkPlots = value
        elif key == 'Dmax':
            Dmax = value
        elif key == 'Dmax2':
            Dmax2 = value
        elif key == 'AutoLdmks':
            AUTO = value
        elif key == 'Overwrite':
            OverwriteData = value
        elif key == 'FirstSlice':
            FirstSlice = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    print('\033[1m' + '\033[4m' + '\nAnalyzing ' + dateCond + ':\n' + '\033[0m')
    
    if doLPR|doL:
        ### Loading area and contour data
        if not os.path.exists(Path + '\\GlobalData' + stringName + '_Landmarks.csv'):
            if not os.path.exists(Path + '\\GlobalData' + stringName + '_Landmarks_tmp.csv'):
                ContourData = pd.read_csv(Path + '\\ContourData' + stringName + '_AreaFit.csv', index_col = 'Ind')
                GlobalData = pd.read_csv(Path + '\\GlobalData' + stringName + '_AreaFit.csv', index_col = 'Ind')
                print('\n Loaded AreaFit file.')
            else:            
                ContourData = pd.read_csv(Path + '\\ContourData' + stringName + '_Landmarks_tmp.csv', index_col = 'Ind')
                GlobalData = pd.read_csv(Path + '\\GlobalData' + stringName + '_Landmarks_tmp.csv', index_col = 'Ind')
                print('\n Loaded Landmarks_tmp file.')
        else:            
            ContourData = pd.read_csv(Path + '\\ContourData' + stringName + '_Landmarks.csv', index_col = 'Ind')
            GlobalData = pd.read_csv(Path + '\\GlobalData' + stringName + '_Landmarks.csv', index_col = 'Ind')
            print('\n Loaded Landmarks file.')
            
        StackList = np.unique(GlobalData.index)

        GlobalData = GlobalData.loc[StackList]   
        ContourData = ContourData.loc[StackList]   

        print('\n\n\nGetting landmarks for : ' + dateCond + '\n\n')
        ContourData_LM, GlobalData_LM = getLandmarks(ContourData,GlobalData,StackList,Scale,Path,stringName, 
                                                     debug = DebugPlots, saveplots = LdmkPlots,
                                                     Dmax = Dmax, Dmax2 = Dmax2, Auto = AUTO, Overwrite = OverwriteData,
                                                     FirstSlice = FirstSlice)
        
        if doLPR:
            # deleting tmp files
            os.remove(Path + '\\GlobalData' + stringName + '_Landmarks_tmp.csv')
            os.remove(Path + '\\ContourData' + stringName + '_Landmarks_tmp.csv')
            GlobalData_LM.to_csv(Path + '\\GlobalData' + stringName + '_Landmarks.csv',index_label = 'Ind')
            ContourData_LM.to_csv(Path + '\\ContourData' + stringName + '_Landmarks.csv',index_label = 'Ind')
            print('\nLandmarks saved.\n\n')

    elif doPR|doR:
        ### Loading landmarks
        GlobalData_LM = pd.read_csv(Path + '\\GlobalData' + stringName + '_Landmarks.csv',index_col = 'Ind')
        ContourData_LM = pd.read_csv(Path + '\\ContourData' + stringName + '_Landmarks.csv',index_col = 'Ind')
        StackList = np.unique(GlobalData_LM.index)

    if doLPR|doPR:
        print('\n\n\nComputing parametric contours for : ' + dateCond + '\n\n')
        ContourData_Param,GlobalData_Param = curvAbsci(ContourData_LM,GlobalData_LM,StackList,Path, debug = DebugPlots)
        ContourData_Param.to_csv(Path + '\\ContourData' + stringName + '_Param.csv',index_label = 'Ind')
        GlobalData_Param.to_csv(Path + '\\GlobalData' + stringName + '_Param.csv',index_label = 'Ind')
        print('\nParametric contours saved.\n\n')

    elif doR:
        ### loading non parametric contours
        GlobalData_Param = pd.read_csv(Path + '\\GlobalData' + stringName + '_Param.csv',index_col = 'Ind')
        ContourData_Param = pd.read_csv(Path + '\\ContourData' + stringName + '_Param.csv',index_col = 'Ind')

    if doLPR|doPR|doR:       
        print('\n\n\nAligning contours for : ' + dateCond + '\n\n')
        ContourData_RC,GlobalData_RC = rotateAndCenterShape(ContourData_Param,GlobalData_Param,StackList,Path,Scale, debug = DebugPlots)
        GlobalData_RC.to_csv(Path + '\\GlobalData' + stringName + '_ParamAligned.csv',index_label = 'Ind')
        ContourData_RC.to_csv(Path + '\\ContourData' + stringName + '_ParamAligned.csv',index_label = 'Ind')
        print('\nAligned parametric contours saved.\n\n')
        
    elif doL:
        print('Landmarks tmp round done for : ' + dateCond)
    else:
        print('No analysis done for : ' + dateCond)
        
    return


#%% Shape quantifications wrapper function

def quantifyShape(ExpName,Pfig,Path,ToDo,**kwargs):
    
    ## Kwargs    
    showPlots = False
    doPlots = False
    DebugPlots = False    
    NormalProj = True    
        
    for key, value in kwargs.items(): 
        if key == 'doPlots':
            doPlots = value
        elif key == 'showPlots':
            showPlots = value  
        elif key == 'DebugPlots':
            DebugPlots = value  
        elif key == 'NormalProj':
            NormalProj = value         
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    
    if 'M' in ToDo:
        
        print('\n\nMean contour computation for expe : ' + ExpName, end =  '\n\n')
    
        ### Loding data of parametrised contours
        CD = pd.read_csv(Path + '\\ContourData' + ExpName + '_ParamAligned.csv', index_col = 'Ind')
        GD = pd.read_csv(Path + '\\GlobalData' + ExpName + '_ParamAligned.csv', index_col = 'Ind') 
        Tstarts = GD.loc[GD['Img']==0,'tdebShift']
        
        ### computation of mean contour
        meanCD,meanGD,GD = computeMeanContourTime(CD,GD,Tstarts)
        
        ### Saving mean contour
        meanGD.to_csv(Path + '\\GlobalData' + ExpName + '_MeanCont.csv',index_label = 'Ind')
        meanCD.to_csv(Path + '\\ContourData' + ExpName + '_MeanCont.csv',index_label = 'Ind')
        
        ### Saving additionnal info
        GD.to_csv(Path + '\\GlobalData' + ExpName + '_ParamAligned.csv',index_label = 'Ind')
        
        del CD, GD, meanCD, meanGD, Tstarts
            
    
    if 'D' in ToDo:
        print('\n\nDistance to mean contour computation for expe : ' + ExpName, end =  '\n\n')
        
        CD = pd.read_csv(Path + '\\ContourData' + ExpName + '_ParamAligned.csv', index_col = 'Ind')
        meanCD = pd.read_csv(Path + '\\ContourData' + ExpName + '_MeanCont.csv', index_col = 'Ind')
        GD = pd.read_csv(Path + '\\GlobalData' + ExpName + '_ParamAligned.csv', index_col = 'Ind') 
        
        ### Compute distance to mean (saved in GD)
        GD = DistToMean(Pfig,CD,GD,meanCD,ExpName,doPlots = doPlots, showPlots = showPlots)
        
        ### Saving
        GD.to_csv(Path + '\\GlobalData' + ExpName + '_DistToMean.csv',index_label = 'Ind')
        
        del CD, GD, meanCD
    
    if 'G' in ToDo:
        print('\n\nContour growth computation for expe : ' + ExpName, end =  '\n\n')
        
        CD = pd.read_csv(Path + '\\ContourData' + ExpName + '_ParamAligned.csv', index_col = 'Ind')
    
        ### Contour growth computation
        GrowthMat = GetGrowth(CD,DebugPlots =DebugPlots,NormalProj = NormalProj)
        
        GrowthMat.to_csv(Path + '\\GrowthData' + ExpName + '.csv',index_label = 'Ind')
        
        del CD, GrowthMat
    
    if 'S' in ToDo:
         
        print('Contour symetry computation for expe : ' + ExpName, end =  '\n\n')
        
        
        CD = pd.read_csv(Path + '\\ContourData' + ExpName + '_ParamAligned.csv', index_col = 'Ind')
        GD = pd.read_csv(Path + '\\GlobalData' + ExpName + '_DistToMean.csv', index_col = 'Ind')
        ### Compute symetry
        GD = computeSym(CD,GD)
        
        ### Final Saving
        GD.to_csv(Path + '\\GlobalData' + ExpName + '_DistToMean_Symetry.csv',index_label = 'Ind')
    
