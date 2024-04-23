#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:12:21 2024

@author: emuller
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
from skimage import io
import cv2 as cv
from PIL import Image
from IPython import get_ipython

from scipy.spatial import ConvexHull
from skimage.morphology import binary_opening, binary_closing, remove_small_holes

import scipy.ndimage
import scipy as sp

from ContourAnalysis import getLandmarks

import VallapFunc as vf

import tifffile as tifff

from GemmaeDetection import create_circular_mask
from OpticalFlow import add_border, add_border_left, rotate,RotateTranslate_Automatic,OpticalFlow_Bckg,RemoveBackgroundStack_Flow

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#%%

def sortGemmae_shocks(RGBimg, s, GD, size = 300):
        
    get_ipython().run_line_magic('matplotlib', 'qt')
    
    f,ax = plt.subplots(dpi=250, ncols = 2)
    ax[0].set_title("Select option for " + str(s), color = 'black', fontsize=15)
    
    AreaC = GD.loc[s,'Area'].values
    Time = GD.loc[s, 'Time (min)'].values
    ax[1].plot(Time,AreaC,'*y',ms=3,label='FullData')
    ax[1].set_xlabel('Time (min)')
    ax[1].set_ylabel('Area (mm²)')
    ax[1].set_title("Times", color = 'black', fontsize=15)
    
    
    rect1 = mpl.patches.Rectangle((5, 50), 30, 30, linewidth=2, edgecolor='r', facecolor='none')
    rect2 = mpl.patches.Rectangle((5, 150), 30, 30, linewidth=2, edgecolor='r', facecolor='none')

    
    # Add the patch to the Axes
    ax[0].add_patch(rect1)
    ax[0].text(45, 70, 'OK', c='r')
    
    ax[0].add_patch(rect2)
    ax[0].text(45, 170, 'Not OK', c='r')
    
    
    ax[0].imshow(RGBimg)
    
    f.tight_layout()
    
    pts = np.asarray(plt.ginput(1, timeout=-1))
    
    plt.close()
    get_ipython().run_line_magic('matplotlib', 'inline')

    if pts[0][1] < 85 and pts[0][1] > 45 and pts[0][0] < 40 and pts[0][0] > 0:
        sort = 'OK'
    elif pts[0][1] < 185 and pts[0][1] > 145 and pts[0][0] < 40 and pts[0][0] > 0:
        sort = 'Not OK'
    else :
        return False, 0
    
    return True, sort

def selectGemmae_shock(P, StackList, GD, size = 300, nimg = 10):
    
    OpticalStack = []
    
    for s in StackList :
        RGBimg = io.imread(P + '/' + s + '.tif', key = nimg)
        RGBimg = add_border_left(RGBimg, size = size)
        Done = False
        
        while not Done :
        
            Done, sort = sortGemmae_shocks(RGBimg, s, GD, size = size)
        
        if sort == 'OK':
           OpticalStack.append(s)        

    return OpticalStack

def SelectTimes( s, GD, size = 300):
        
    get_ipython().run_line_magic('matplotlib', 'qt')
    
    f,ax = plt.subplots(dpi=250)
    
    AreaC = GD.loc[s,'Area'].values
    Time = GD.loc[s, 'Time (min)'].values
    ax.plot(Time,AreaC,'*y',ms=3,label='FullData')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Area (mm²)')
    ax.set_title("Slect Times", color = 'black', fontsize=15)
   
    pts = np.asarray(plt.ginput(4, timeout=-1))
    
    plt.close()
    get_ipython().run_line_magic('matplotlib', 'inline')

    
    return [p[0] for p in pts]


#%%

def MainOpticalFlow_AutomaticLandmarks_RemoveBgLast_shocks(P, StackList, Scale, ToDo, stringName, **kwargs):
    
    # keyword arguments to find reference point
    overwriteRefs = False
    
    # keyword arguments for rotate/translate image
    factorA = [0.2]*len(StackList)
    DebugPlotsA = False
    
    
    ImgList = [0, 20, 40]
    
    # keyword argument for remove background
    DebugPlotsB = False                            
    HSVmin = (30, 20, 70)
    HSVmax = (60, 120,220)
    saveWB = False
    factor1 = 2
    factor2 = 30
    Areath = 5*1e3
    DebugPlotsFlow = False
    
    # keyword arguments for optical flow
    to_gray = True
    
    for key, value in kwargs.items(): 
        if key == 'overwriteRefs':
            overwriteRefs = value
        elif key == 'factorA':
            factorA = value
        elif key == 'DebugPlotsA':
            DebugPlotsA = value
        elif key == 'ImgList':
            ImgList= value
        elif key == 'DebugPlotsB':
            DebugPlotsB = value
        elif key == 'saveWB':
            saveWB = value 
        elif key == 'HSVrange':
            HSVmin = value[0]
            HSVmax = value[1]
        elif key == "Binfactor1":
            factor1 = value
        elif key == "Binfactor2":
            factor2 = value
        elif key == "Binthreshold" :
            Areath = value
        elif key == "to_gray":
            to_gray = value
        elif key == 'DebugPlotsFlow':
            DebugPlotsFlow = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    DoSelect = False
    DoAlign = False
    DoBackground = False
    DoOpticalFlow = False
    
    if "S" in ToDo:
        DoSelect = True
    if "A" in ToDo:
        DoAlign = True
    if "B" in ToDo:
        DoBackground = True
    if "O" in ToDo :
        DoOpticalFlow = True
    
    if DoSelect:
        GlobalData = pd.read_csv(P + '/GlobalData' + stringName + '_AreaFit.csv', index_col = 'Ind')
        StackList = [s for s in StackList if (s in np.unique(GlobalData.index.values)) ]
        OpticalStack = selectGemmae_shock(P, StackList, GlobalData, size = 300, nimg = 10)
        
        np.savetxt(P+"/OpticalStack.csv",OpticalStack,delimiter =", ",fmt ='% s')
        
        if not os.path.exists(P + '/Reduced/'):
            os.mkdir(P + '/Reduced/') # create binarize image folder
        
        WB = pd.read_csv(P+'/WhiteBalanceData.csv', index_col = 'Ind')   
        WB.to_csv(P+'/Reduced/WhiteBalanceData.csv', index_label = 'Ind')  
        
        GemmaeTime = pd.DataFrame(columns = ['T0', 'T100mM', 'T250mM', 'TPlasmo'])
        ContourData = pd.read_csv(P + '/ContourData' + stringName + '_AreaFit.csv', index_col = 'Ind')
        GlobalData = pd.read_csv(P + '/GlobalData' + stringName + '_AreaFit.csv', index_col = 'Ind')
        
        for s in OpticalStack :
            Times = SelectTimes( s, GlobalData, size = 300)
            Times = [round(t*2)/2 for t in Times]
            GemmaeTime = pd.concat([GemmaeTime, pd.DataFrame([Times],columns = ['T0', 'T100mM', 'T250mM', 'TPlasmo'], index = [s])])
            
            # create reduce image dataset
            Nimg = [t*2 for t in Times]
            img = io.imread(P+s+'.tif', key = Nimg)
            io.imsave(P + '/Reduced/' +s + '.tif', img, plugin='tifffile')
            ContourData = ContourData.loc[(ContourData.index != s) | (ContourData['Img'] == Nimg[0]) | (ContourData['Img'] == Nimg[1]) | (ContourData['Img'] == Nimg[2]) | (ContourData['Img'] == Nimg[3]), ]
            GlobalData = GlobalData.loc[(GlobalData.index != s) | (GlobalData['Time (min)'] == Times[0]) | (GlobalData['Time (min)'] == Times[1]) | (GlobalData['Time (min)'] == Times[2]) | (GlobalData['Time (min)'] == Times[3]), ]
            
            i = 0
            for ind in np.unique(ContourData.loc[s, 'Img']):
                ContourData.loc[(ContourData.index == s) & (ContourData['Img'] == ind), 'Img'] = i
                GlobalData.loc[(GlobalData.index == s) & (GlobalData['Img'] == ind), 'Img'] = i
                i += 1
            
        
        GemmaeTime.to_csv(P+'/Reduced/GemmaeTimes.csv',index_label = 'Ind')
        ContourData.to_csv(P + '/Reduced/ContourData' + stringName + '_AreaFit.csv',index_label = 'Ind')
        GlobalData.to_csv(P + '/Reduced/GlobalData' + stringName + '_AreaFit.csv',index_label = 'Ind')
    
    else :
        OpticalStack = np.loadtxt(P+"/OpticalStack.csv", delimiter =", ", dtype = str)
        GemmaeTime = pd.read_csv(P+'/Reduced/GemmaeTimes.csv')
        #print(OpticalStack)
        
    if "UsePPG" in ToDo:
        OpticalStack = StackList
        
     
    P = P+'/Reduced/'
    
    if DoAlign:
        
        if not os.path.exists(P + '/GlobalData' + stringName + '_Landmarks.csv'):
            if not os.path.exists(P + '/GlobalData' + stringName + '_Landmarks_tmp.csv'):
                ContourData = pd.read_csv(P + '/ContourData' + stringName + '_AreaFit.csv', index_col = 'Ind')
                GlobalData = pd.read_csv(P + '/GlobalData' + stringName + '_AreaFit.csv', index_col = 'Ind')
                print('\n Loaded AreaFit file.')
            else:            
                ContourData = pd.read_csv(P + '/ContourData' + stringName + '_Landmarks_tmp.csv', index_col = 'Ind')
                GlobalData = pd.read_csv(P + '/GlobalData' + stringName + '_Landmarks_tmp.csv', index_col = 'Ind')
                print('\n Loaded Landmarks_tmp file.')
        else:            
            ContourData = pd.read_csv(P + '/ContourData' + stringName + '_Landmarks.csv', index_col = 'Ind')
            GlobalData = pd.read_csv(P + '/GlobalData' + stringName + '_Landmarks.csv', index_col = 'Ind')
            print('\n Loaded Landmarks file.')
        
        ContourData_LM, GlobalData_LM = getLandmarks(ContourData,GlobalData,OpticalStack,Scale,P,stringName, FirstSlice=False, Overwrite = overwriteRefs)
        
        if os.path.exists(P + '/GlobalData' + stringName + '_Landmarks_aligned.csv'):
            GlobalData_LM = pd.read_csv(P + '/GlobalData' + stringName + '_Landmarks_aligned.csv', index_col = 'Ind')
            
        RotateTranslate_Automatic(GlobalData_LM, OpticalStack,P, stringName, factor = factorA, DebugPlots = DebugPlotsA, ImgList = ImgList)
        
        
    if DoOpticalFlow:
    
        if not os.path.exists(P + '/Flow'):
                os.mkdir(P + '/Flow') 
    
        for s in OpticalStack:
    
            print('Processing ' + s + ' :')  
            
            if not os.path.exists(P + '/Flow/' + s + '_Flow'):
                os.mkdir(P + '/Flow/' + s + '_Flow')
            
            OpticalFlow_Bckg(P, s, to_gray=to_gray)
            
    if DoBackground:
        RemoveBackgroundStack_Flow(OpticalStack, P, Scale, DebugPlots = DebugPlotsB, ImgList = ImgList,saveWB = saveWB, HSVrange = [HSVmin, HSVmax], Binfactor1 = factor1, Binfactor2 = factor2, Binthreshold = Areath, DebugPlotsFlow = DebugPlotsFlow)
    
    return