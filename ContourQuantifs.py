# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 09:12:58 2022

@author: laplaud
"""

from ContourAnalysis import RotTransAlignement

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os

import VallapFunc as vf

#%% Computation of mean contours
# With and without alignement in time based on dormancy exit 

# Contours computed when at least 20 gemmae are averageable

def computeMeanContourTime(CD,GD,Tstarts):
    StackList = np.unique(Tstarts.index)
    
    meanCD = pd.DataFrame(data=None,columns = ['X','Y','Img'])
    meanGD = pd.DataFrame(data=None,columns = ['nppgs','Img'])
    
    # number of contour points
    npts = len(CD.loc[(CD.index == StackList[0]) & (CD['Img'] == 0),'X'])
    
############################## Contours averaged at each time point ###################################################    
    cnt = 0 # number of mean contours done
    
    SL = StackList[:]
    
    nppgs = len(SL)
    
    while nppgs>19:
        
        print('Computing mean contour n°' + str(cnt+1) + ' :')
        
        # Storing variables
        Xs = np.empty((npts,nppgs))
        Xs[:] = np.nan
        Ys = np.empty((npts,nppgs))
        Ys[:] = np.nan


        for s,i in zip(SL,range(len(SL))) :


            Xs[:,i] = CD.loc[(CD.index==s) & (CD['Img'] == int(cnt)), 'X']
            Ys[:,i] = CD.loc[(CD.index==s) & (CD['Img'] == int(cnt)), 'Y']
            

        meanX = np.mean(Xs,axis=1)
        meanY = np.mean(Ys,axis=1)           

        print(str(nppgs) + ' contours averaged together.\n')

        data = {'X':meanX,
                    'Y':meanY,
                    'Img':cnt*np.ones(len(meanX))} 

        newdata = pd.DataFrame(data=data,index = np.repeat('meanCnotal',len(meanX)))

        meanCD = meanCD.append(newdata) # adding to global dataframe
        
        data = {'nppgs':nppgs,
            'Img':cnt} 

        newdata = pd.DataFrame(data=data,index = ['meanCnotal'])
        
        meanGD = meanGD.append(newdata)
        
        
        cnt = cnt+1
        newSL = []
        for s in SL :
            if np.max(CD.loc[s,'Img'])-cnt>0:
                newSL.append(s)
        
        SL = newSL[:]
        
        nppgs = len(SL)
        

 ################################ Contours aligned in time based on dormancy exit ####################################
    cnt = 0 # number of mean contours done
    
    SL = StackList[:]
    
    nppgs = len(SL)

    while nppgs>19:

        print('Computing mean contour (aligned in time) n°' + str(cnt+1) + ' :')

        # Storing variables
        Xs = np.empty((npts,nppgs))
        Xs[:] = np.nan
        Ys = np.empty((npts,nppgs))
        Ys[:] = np.nan


        for s,i in zip(SL,range(len(SL))) :

            Xs[:,i] = CD.loc[(CD.index==s) & (CD['Img'] == int(cnt+Tstarts[s])), 'X']
            Ys[:,i] = CD.loc[(CD.index==s) & (CD['Img'] == int(cnt+Tstarts[s])), 'Y']
            
            GD.loc[(GD.index==s) & (GD['Img'] == int(cnt+Tstarts[s])),'MeanCimg'] = cnt

        meanX = np.mean(Xs,axis=1)
        meanY = np.mean(Ys,axis=1)           

        print(str(nppgs) + ' contours averaged together.\n')

        data = {'X':meanX,
                    'Y':meanY,
                    'Img':cnt*np.ones(len(meanX))} 

        newdata = pd.DataFrame(data=data,index = np.repeat('meanC',len(meanX)))

        meanCD = meanCD.append(newdata) # adding to global dataframe
        
        data = {'nppgs':nppgs,
            'Img':cnt} 

        newdata = pd.DataFrame(data=data,index = ['meanC'])
        
        meanGD = meanGD.append(newdata)
        
        
        cnt = cnt+1
        newSL = []
        for s in SL :
            if np.max(CD.loc[s,'Img'])-Tstarts[s]-cnt>0:
                newSL.append(s)
        
        SL = newSL[:]
        
        nppgs = len(SL)
    
    return(meanCD,meanGD)
            

#%% Distance to mean contour for each ppg

def DistToMean(Pfig,CD,GD,CDmean,Expname,**kwargs):
    
    ## Params
    StackList = np.unique(GD.index)
    
    ## Kwargs    
    showPlots = False
    doPlots = False
    
    for key, value in kwargs.items(): 
        if key == 'doPlots':
            doPlots = value
            showPlots = value
        elif key == 'showPlots':
            showPlots = value         
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    ## Saving folders
    if not os.path.exists(Pfig + '\\MeanContoursVariability\\'):
        os.mkdir(Pfig + '\\MeanContoursVariability\\') # create general folder  
    
    fullP = Pfig + '\\MeanContoursVariability\\' + Expname + '\\'
    
    if not os.path.exists(fullP):
            os.mkdir(fullP) # create experiment folder 
    
    if not os.path.exists(fullP + '\\OriginalContours\\'):
            os.mkdir(fullP + '\\OriginalContours\\') # create experiment folder  
    
    if not os.path.exists(fullP + '\\NormalizedContours\\'):
            os.mkdir(fullP + '\\NormalizedContours\\') # create experiment folder    
              
############################## Contours averaged at each time point ###################################################      
    nimg = int(np.max(CDmean.loc['meanCnotal','Img']))
    
    for i in range(nimg+1):
        
        print('Computing variability around mean contours (not time aligned) n°' + str(i))
        
        Xmean = CDmean.loc[(CDmean.index == 'meanCnotal') & (CDmean['Img']==i),'X'].values
        Ymean = CDmean.loc[(CDmean.index == 'meanCnotal') & (CDmean['Img']==i),'Y'].values
        
        # normalized mean contour
        XmeanN,YmeanN = vf.normaliseContourByArea(Xmean,Ymean)
        
        if doPlots:                        
            fig1,ax1 = plt.subplots(dpi=300)
            fig1.suptitle(Expname + ' contours+mean (not time aligned) img n°' + str(i))
            ax1.set_aspect('equal', adjustable='box')
            ax1.set_xlabel('X (µm)')
            ax1.set_ylabel('Y (µm)')
            
            
            boundX = [np.max(CDmean.loc[(CDmean.index == 'meanCnotal') & (CDmean['Img']==i),'X'])*1.5,
                      np.min(CDmean.loc[(CDmean.index == 'meanCnotal') & (CDmean['Img']==i),'X'])*1.5]
            boundY = [np.max(CDmean.loc[(CDmean.index == 'meanCnotal') & (CDmean['Img']==i),'Y'])*1.5,
                      np.min(CDmean.loc[(CDmean.index == 'meanCnotal') & (CDmean['Img']==i),'Y'])*1.5]
            
            ax1.plot(boundX,boundY,'ko',ms = 1,zorder = -2)
            
            ax1.plot(Xmean,Ymean,'r-',lw= 2,zorder=0)
            
            fig2,ax2 = plt.subplots(dpi=300)
            fig2.suptitle(Expname + ' normalized contours+mean (not time aligned) img n°' + str(i))
            ax2.set_aspect('equal', adjustable='box')
            ax2.set_xlabel('X (norm)')
            ax2.set_ylabel('Y (norm)') 
            ax2.set_xlim(-0.8,0.8)
            ax2.set_ylim(-0.8,0.8)
        
            ax2.plot(XmeanN,YmeanN,'r-',lw= 2,zorder=0)
        
        for s in StackList:
            
            if not CD.loc[ (CD.index==s) & (CD['Img'] == i ),'X'].values.size==0:                
                
                X = CD.loc[ (CD.index==s) & (CD['Img'] == i ),'X'].values
                Y = CD.loc[ (CD.index==s) & (CD['Img'] == i ),'Y'].values
                
                AllDists = vf.dist(X,Y,Xmean,Ymean)
                MedDist = np.nanmedian(AllDists);
                L2Dist = np.sqrt(np.sum(np.square(AllDists)))/np.sqrt(len(AllDists))
                
                GD.loc[(GD.index==s) & (GD['Img']==i),'NotalMedDist'] = MedDist
                GD.loc[(GD.index==s) & (GD['Img']==i),'NotalL2Dist'] = L2Dist
                
                # normalized mean contour
                XN,YN = vf.normaliseContourByArea(X,Y)
                
                AllDistsN = vf.dist(XN,YN,XmeanN,YmeanN)
                MedDistN = np.nanmedian(AllDistsN);
                L2DistN = np.sqrt(np.sum(np.square(AllDistsN)))/np.sqrt(len(AllDistsN))
                
                GD.loc[(GD.index==s) & (GD['Img']==i),'NotalMedDistNorm'] = MedDistN
                GD.loc[(GD.index==s) & (GD['Img']==i),'NotalL2DistNorm'] = L2DistN
            
                if doPlots:
                    ax1.plot(X,Y,'w-',lw=0.3,zorder=-1)
                    ax2.plot(XN,YN,'w-',lw=0.3,zorder=-1)
        
        
        if doPlots:
            fig1.savefig(fullP + '\\OriginalContours\\' + 'NotalMeanAndVar_Img' + str(i)  + '.png')
            fig2.savefig(fullP + '\\NormalizedContours\\' + 'Norm_NotalMeanAndVar_Img' + str(i)  + '.png')
            
        
            if showPlots:
                plt.show() 
            else:
                plt.close(fig1)
                plt.close(fig2)
                
                
 ################################ Contours aligned in time based on dormancy exit ####################################    
    nimg = int(np.max(CDmean.loc['meanC','Img']))
    
    for i in range(nimg+1):
        
        print('Computing variability around mean contours n°' + str(i))
        
        Xmean = CDmean.loc[(CDmean.index == 'meanC') & (CDmean['Img']==i),'X'].values
        Ymean = CDmean.loc[(CDmean.index == 'meanC') & (CDmean['Img']==i),'Y'].values
        
        # normalized mean contour
        XmeanN,YmeanN = vf.normaliseContourByArea(Xmean,Ymean)
        
        if doPlots:                        
            fig1,ax1 = plt.subplots(dpi=300)
            fig1.suptitle(Expname + ' contours+mean img n°' + str(i))
            ax1.set_aspect('equal', adjustable='box')
            ax1.set_xlabel('X (µm)')
            ax1.set_ylabel('Y (µm)')
            
            
            boundX = [np.max(CDmean.loc[(CDmean.index == 'meanC') & (CDmean['Img']==i),'X'])*1.5,
                      np.min(CDmean.loc[(CDmean.index == 'meanC') & (CDmean['Img']==i),'X'])*1.5]
            boundY = [np.max(CDmean.loc[(CDmean.index == 'meanC') & (CDmean['Img']==i),'Y'])*1.5,
                      np.min(CDmean.loc[(CDmean.index == 'meanC') & (CDmean['Img']==i),'Y'])*1.5]
            
            ax1.plot(boundX,boundY,'ko',ms = 1,zorder = -2)
            
            ax1.plot(Xmean,Ymean,'r-',lw= 2,zorder=0)
            
            fig2,ax2 = plt.subplots(dpi=300)
            fig2.suptitle(Expname + ' normalized contours+mean img n°' + str(i))
            ax2.set_aspect('equal', adjustable='box')
            ax2.set_xlabel('X (norm)')
            ax2.set_ylabel('Y (norm)') 
            ax2.set_xlim(-0.8,0.8)
            ax2.set_ylim(-0.8,0.8)
        
            ax2.plot(XmeanN,YmeanN,'r-',lw= 2,zorder=0)
        
        for s in StackList:
            
            if not GD.loc[(GD.index==s)&(GD['MeanCimg']==i),'Img'].values.size==0:
                
                AlgImg = GD.loc[ (GD.index==s) & (GD['MeanCimg']==i),'Img'].values[0]
                
                X = CD.loc[ (CD.index==s) & (CD['Img'] == AlgImg ),'X'].values
                Y = CD.loc[ (CD.index==s) & (CD['Img'] == AlgImg ),'Y'].values
                
                AllDists = vf.dist(X,Y,Xmean,Ymean)
                MedDist = np.nanmedian(AllDists);
                L2Dist = np.sqrt(np.sum(np.square(AllDists)))/len(AllDists)
                
                GD.loc[(GD.index==s) & (GD['MeanCimg']==i),'MedDist'] = MedDist
                GD.loc[(GD.index==s) & (GD['MeanCimg']==i),'L2Dist'] = L2Dist
                
                # normalized mean contour
                XN,YN = vf.normaliseContourByArea(X,Y)
                
                AllDistsN = vf.dist(XN,YN,XmeanN,YmeanN)
                MedDistN = np.nanmedian(AllDistsN);
                L2DistN = np.sqrt(np.sum(np.square(AllDistsN)))/len(AllDistsN)
                
                GD.loc[(GD.index==s) & (GD['MeanCimg']==i),'MedDistNorm'] = MedDistN
                GD.loc[(GD.index==s) & (GD['MeanCimg']==i),'L2DistNorm'] = L2DistN
            
                if doPlots:
                    ax1.plot(X,Y,'w-',lw=0.3,zorder=-1)
                    ax2.plot(XN,YN,'w-',lw=0.3,zorder=-1)
        
        
        if doPlots:
            fig1.savefig(fullP + '\\OriginalContours\\' + 'MeanAndVar_Img' + str(i)  + '.png')
            fig2.savefig(fullP + '\\NormalizedContours\\' + 'Norm_MeanAndVar_Img' + str(i)  + '.png')
            
        
            if showPlots:
                plt.show() 
            else:
                plt.close(fig1)
                plt.close(fig2)
            
    