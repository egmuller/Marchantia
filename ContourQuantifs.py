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
            
    
    
#%% Contour growth

def GetGrowth(CD,**kwargs):
    
    DebugPlots = False    
    NormalProj = True
    
    for key, value in kwargs.items(): 
        if key == 'DebugPlots':
            DebugPlots = value
        elif key == 'NormalProj':
            NormalProj = value         
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    StackList = np.unique(CD.index)
    
    GrowthData = pd.DataFrame(data=None,columns = ['InstantGrowth','TotGrowth','AccumulatedGrowth','Img'])
    
    for s in StackList:
        
        nimg = int(np.max(CD.loc[s, 'Img']))
#         nimg=3
        
        X0 = CD.loc[(CD.index == s) & (CD['Img'] == 0),'X'].values 
        Y0 = CD.loc[(CD.index == s) & (CD['Img'] == 0),'Y'].values
        
        contourVectors_0 = np.array([[x2-x1,y2-y1] for (x1,y1,x2,y2) in zip(np.roll(X0,+1),np.roll(Y0,+1),np.roll(X0,-1),np.roll(Y0,-1))])   

        normalVectors_0 = np.array([vf.getNormal(0,0,x,y,rotation='CCW') for (x,y) in contourVectors_0])
        
        DistAccum = np.zeros(len(X0))
        
        for i in range(1,nimg):
            
            print('Computing growth for ' + s + ' (img : ' + str(i) + ')', end='\r')

            Xi = np.array(CD.loc[(CD.index == s) & (CD['Img'] == i),'X'].values)
            Yi = np.array(CD.loc[(CD.index == s) & (CD['Img'] == i),'Y'].values)
            
            Xi_1 = np.array(CD.loc[(CD.index == s) & (CD['Img'] == i-1),'X'].values)
            Yi_1 = np.array(CD.loc[(CD.index == s) & (CD['Img'] == i-1),'Y'].values)
            
            growthVectors = np.array([[xi-xi_1,yi-yi_1] for (xi,xi_1,yi,yi_1) in zip(Xi,Xi_1,Yi,Yi_1)])
            
            totGrowthVectors = np.array([[xi-x0,yi-y0] for (xi,x0,yi,y0) in zip(Xi,X0,Yi,Y0)])

            contourVectors = np.array([[x2-x1,y2-y1] for (x1,y1,x2,y2) in zip(np.roll(Xi_1,+1),np.roll(Yi_1,+1),np.roll(Xi_1,-1),np.roll(Yi_1,-1))])   

            normalVectors = np.array([vf.getNormal(0,0,x,y,rotation='CCW') for (x,y) in contourVectors])
              
            if NormalProj:
                DistInst = np.array([np.dot(g,n) for (g,n) in zip(growthVectors,normalVectors)])
                DistTot = np.array([np.dot(g,n) for (g,n) in zip(totGrowthVectors,normalVectors_0)])
                DistAccum = np.sum((DistAccum,DistInst),axis=0)
            else:
                DistInst = np.array([np.linalg.norm(g) for g in growthVectors])
                DistTot = np.array([np.linalg.norm(g) for g in totGrowthVectors])
                DistAccum = np.sum((DistAccum,DistInst),axis=0)


            if DebugPlots:
                
                    pos = [991,992,993,994,995,996,997,998,999,0,1,2,3,4,5,6,7,8,9,10,11,12]
                    pos1 = np.linspace(420,450).astype(int)
                    pos2 = np.linspace(730,760).astype(int)
                
                    fig, ax = plt.subplots(ncols = 2,nrows=2,dpi=300)
                    fig.suptitle('Growth between image ' + str(i-1) + ' & ' + str(i+10))

                    ax[0][0].plot(Xi,Yi,'-g',lw=.5) 
                    ax[0][0].plot(Xi_1,Yi_1,'-b',lw=.5) 
                    ax[0][1].plot(Xi[pos],Yi[pos],'-g',lw=.5)  
                    ax[1][0].plot(Xi[pos1],Yi[pos1],'-g',lw=.5)  
                    ax[1][1].plot(Xi[pos2],Yi[pos2],'-g',lw=.5)   

                    for [x,y],xi_1,yi_1 in zip(contourVectors[pos],np.roll(Xi_1[pos],+1),np.roll(Yi_1[pos],+1)):
                        ax[0][1].quiver(xi_1,yi_1,x,y,color='b',scale=1,scale_units='xy')

                    for [x,y],xi_1,yi_1 in zip(growthVectors[pos],Xi_1[pos],Yi_1[pos]):
                        ax[0][1].quiver(xi_1,yi_1,x,y,color='r',scale=1,scale_units='xy')
                        ax[0][0].quiver(xi_1,yi_1,x,y,color='r',scale=1,scale_units='xy',zorder=5)

                    for [x,y],xi_1,yi_1,d in zip(normalVectors[pos],Xi_1[pos],Yi_1[pos],DistInst[pos]):
                        ax[0][1].quiver(xi_1,yi_1,x,y,color='c',scale=1,scale_units='xy',zorder=5)
                        ax[0][1].plot([xi_1,xi_1+d*x],[yi_1,yi_1+d*y],'-m',lw=.5)
                        

                        
                    for [x,y],xi_1,yi_1 in zip(contourVectors[pos1],np.roll(Xi_1[pos1],+1),np.roll(Yi_1[pos1],+1)):
                        ax[1][0].quiver(xi_1,yi_1,x,y,color='b',scale=1,scale_units='xy')

                    for [x,y],xi_1,yi_1 in zip(growthVectors[pos1],Xi_1[pos1],Yi_1[pos1]):
                        ax[1][0].quiver(xi_1,yi_1,x,y,color='r',scale=1,scale_units='xy')
                        ax[0][0].quiver(xi_1,yi_1,x,y,color='r',scale=1,scale_units='xy',zorder=5)

                    for [x,y],xi_1,yi_1,d in zip(normalVectors[pos1],Xi_1[pos1],Yi_1[pos1],DistInst[pos1]):
                        ax[1][0].quiver(xi_1,yi_1,x,y,color='c',scale=1,scale_units='xy')
                        ax[1][0].plot([xi_1,xi_1+d*x],[yi_1,yi_1+d*y],'-m',lw=.5)
                        

                        
                    for [x,y],xi_1,yi_1 in zip(contourVectors[pos2],np.roll(Xi_1[pos2],+1),np.roll(Yi_1[pos2],+1)):
                        ax[1][1].quiver(xi_1,yi_1,x,y,color='b',scale=1,scale_units='xy')

                    for [x,y],xi_1,yi_1 in zip(growthVectors[pos2],Xi_1[pos2],Yi_1[pos2]):
                        ax[1][1].quiver(xi_1,yi_1,x,y,color='r',scale=1,scale_units='xy')
                        ax[0][0].quiver(xi_1,yi_1,x,y,color='r',scale=1,scale_units='xy',zorder=5)

                    for [x,y],xi_1,yi_1,d in zip(normalVectors[pos2],Xi_1[pos2],Yi_1[pos2],DistInst[pos2]):
                        ax[1][1].quiver(xi_1,yi_1,x,y,color='c',scale=1,scale_units='xy')
                        ax[1][1].plot([xi_1,xi_1+d*x],[yi_1,yi_1+d*y],'-m',lw=.5)

                        
                    ax[0][1].plot(Xi_1[pos],Yi_1[pos], 'og',ms=.5)
                    ax[0][1].plot(Xi[pos],Yi[pos],'og',ms=.5)    
                    ax[1][0].plot(Xi_1[pos1],Yi_1[pos1], 'og',ms=.5)
                    ax[1][0].plot(Xi[pos1],Yi[pos1],'og',ms=.5)    
                    ax[1][1].plot(Xi_1[pos2],Yi_1[pos2], 'og',ms=.5)
                    ax[1][1].plot(Xi[pos2],Yi[pos2],'og',ms=.5)    
                    ax[0][0].set_aspect('equal')
                    ax[0][1].set_aspect('equal')  
                    ax[1][0].set_aspect('equal')
                    ax[1][1].set_aspect('equal')
                    
                    fig.tight_layout()
            

            data = {'InstantGrowth':DistInst,
                    'TotGrowth' : DistTot,
                    'AccumulatedGrowth' : DistAccum,
                    'Img':i} 

            newdata = pd.DataFrame(data=data,index = np.repeat(s,len(DistInst)))
            
            GrowthData = GrowthData.append(newdata)
                  
        print('Computing growth for ' + s + ' (DONE)'.ljust(20),end='\n')
            
        
    return(GrowthData)
            

#%% Contour symetry

def computeSym(CD,GD):
    StackList = np.unique(CD.index)
    
    for s in StackList:
        
        nimg = int(np.max(CD.loc[s, 'Img']))+1
#         nimg = 1
        
        sym = np.empty(nimg)
        symN = np.empty(nimg)
        symNotch = np.empty(nimg)
        symNotchN = np.empty(nimg)
        
        for i in range(nimg):
            
            print('Computing symmetry for ' + s + ' (img : ' + str(i) + ')', end='\r')

            Xi = CD.loc[(CD.index == s) & (CD['Img'] == i),'X'].values 
            Yi = CD.loc[(CD.index == s) & (CD['Img'] == i),'Y'].values
            
            XiN,YiN = vf.normaliseContourByArea(Xi,Yi)
            
            XiL = np.concatenate((Xi[750:1000],Xi[0:250]))
            YiL = np.concatenate((Yi[750:1000],Yi[0:250]))
            
            XiR = Xi[250:750]
            YiR = Yi[250:750]
            
            XiRr = np.flip(-XiR)
            YiRr = np.flip(YiR)
            
            XiLna = XiL - XiL[250]
            YiLna = YiL - YiL[250]
            
            XiRna = XiRr - XiRr[250]
            YiRna = YiRr - YiRr[250]
            
            XiRda,YiRda = RotTransAlignement(XiRr[:],YiRr[:],XiL[:],YiL[:]
                                          ,np.linspace(-20,20,81),np.linspace(-20,20,81),np.linspace(-20,20,81),False)
            
            
#             fig,ax = plt.subplots(dpi= 300)
#             ax.set_title('Original contour')
#             ax.plot(XiL,YiL,'-b',lw=1)
#             ax.plot(XiR,YiR,'-r',lw=1)
#             ax.set_aspect('equal')
            
#             fig, [ax2,ax3] = plt.subplots(ncols = 2, dpi = 300)
            
#             ax2.set_title('Notch Aligned')
#             ax2.plot(XiLna,YiLna,'-b',lw=1)
#             ax2.plot(XiRna,YiRna,'-r',lw=1)
#             ax2.plot([np.transpose(XiLna[0:500:25]),np.transpose(XiRna[0:500:25])],
#                      [np.transpose(YiLna[0:500:25]),np.transpose(YiRna[0:500:25])],'o-w',ms=2,lw=1.3)
#             ax2.set_aspect('equal')
            
#             ax3.set_title('Distance aligned')
#             ax3.plot(XiL,YiL,'-b',lw=1)
#             ax3.plot(XiRda,YiRda,'-r',lw=1)
#             ax3.plot([np.transpose(XiL[0:500:25]),np.transpose(XiRda[0:500:25])],
#                      [np.transpose(YiL[0:500:25]),np.transpose(YiRda[0:500:25])],'o-w',ms=2,lw=1.3)
#             ax3.set_aspect('equal')
            
#             fig.tight_layout()
            
#             plt.show()
            

            sym[i] = np.median(vf.dist(XiL,YiL,XiRda,YiRda))
            
            symNotch[i] = np.median(vf.dist(XiLna,YiLna,XiRna,YiRna))
            
            
            
            XiLN = np.concatenate((XiN[750:1000],XiN[0:250]))
            YiLN = np.concatenate((YiN[750:1000],YiN[0:250]))
            
            XiRN = XiN[250:750]
            YiRN = YiN[250:750]
            
            XiRrN = np.flip(-XiRN)
            YiRrN = np.flip(YiRN)
            
            XiLnaN = XiLN - XiLN[250]
            YiLnaN = YiLN - YiLN[250]
            
            XiRnaN = XiRrN - XiRrN[250]
            YiRnaN = YiRrN - YiRrN[250]
            
            XiRdaN,YiRdaN = RotTransAlignement(XiRrN[:],YiRrN[:],XiLN[:],YiLN[:]
                                          ,np.linspace(-20,20,81),np.linspace(-20,20,81),np.linspace(-20,20,81),False)
            
            symN[i] = np.median(vf.dist(XiLN,YiLN,XiRdaN,YiRdaN))
            
            symNotchN[i] = np.median(vf.dist(XiLnaN,YiLnaN,XiRnaN,YiRnaN))
            
            GD.loc[(GD.index == s) & (GD['Img'] == i),'Sym'] = sym[i]
            GD.loc[(GD.index == s) & (GD['Img'] == i),'SymNotch'] = symNotch[i]
            GD.loc[(GD.index == s) & (GD['Img'] == i),'SymNorm'] = symN[i]
            GD.loc[(GD.index == s) & (GD['Img'] == i),'SymNotchNorm'] = symNotchN[i]
         
        print('Computing symmetry for ' + s + ' (DONE)'.ljust(20), end='\n')
        

            
            
    