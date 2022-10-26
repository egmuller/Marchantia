# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:15:32 2022

@author: laplaud
"""

import os
import gc

import pandas as pd
import numpy as np
import numpy.matlib as mtl
import seaborn as sns

from scipy import optimize
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d

from skimage import io

import VallapFunc as vf

from matplotlib import path as mpltPath
from matplotlib import pyplot as plt

import time

# other stuff
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


#%% rotateAndCenterShape

def rotateAndCenterShape(CD,GD,StackList,P,Scale, **kwargs):
    
    if not os.path.exists(P + '\\Figures\\Rotation\\'):
        os.mkdir(P + '\\Figures\\Rotation\\') # create folder
    
    DebugPlots = False
    SavedPlots = False
    
    for key, value in kwargs.items(): 
        if key == 'debug':
            DebugPlots = value
        elif key == 'saveplots':
            SavedPlots = value        
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    newCD = pd.DataFrame(data=None,columns=['Img','X','Y','S']) 
    T0X = np.empty((1000,len(StackList)))
    T0Y = np.empty((1000,len(StackList)))

    ## Alignement of all PPG at t = 0
    
    # 1. aligned all first time based on notches
    
    for s,ii in zip(StackList,range(len(StackList))):
        
        print('Aligning first image on notches for ' + s, end='\r')
            
        if not os.path.exists(P + '\\Figures\\Rotation\\'+ s + '\\'):
            os.mkdir(P + '\\Figures\\Rotation\\'+ s + '\\') # create folder

        i = 0 #  for first image

        # retrieve important points and contours
        Xc = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xcenter'].values
        Yc = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Ycenter'].values

        Xparam = CD.loc[(CD.index == s) & (CD['Img'] == i), 'Xparam'].values-Xc
        Yparam = CD.loc[(CD.index == s) & (CD['Img'] == i), 'Yparam'].values-Yc

        XA = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xattach'].values-Xc
        YA = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Yattach'].values-Yc

        Xn1 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xnotch1'].values-Xc
        Yn1 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Ynotch1'].values-Yc

        Xn2 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xnotch2'].values-Xc
        Yn2 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Ynotch2'].values-Yc

        ## align based on the notches position

        A,R = vf.ToCirc([XA,Xn1,Xn2],[YA,Yn1,Yn2],angle='deg')
        Theta,Radius = vf.ToCirc(Xparam,Yparam,angle='deg')

        Anotches = np.mod(A[1:3].T-np.array([0,180]),360)

        Abis = np.mod(Anotches.sum()/2,360)-180

        [XAal,Xn1al,Xn2al],[YAal,Yn1al,Yn2al] = vf.ToCart(A-Abis,R,angle='deg')

        if YAal > 0:
            Abis = Abis + 180

        XparamAlnotch,YparamAlnotch = vf.ToCart(Theta-Abis,Radius,angle='deg')

        T0X[:,ii] = XparamAlnotch
        T0Y[:,ii] = YparamAlnotch
    
    print('First images aligned on notches' + '.'.ljust(20), flush=True)
    
    
    # 2. Compute temporary mean contour
    print('Computing temp mean contour')
    MeanT0X  = T0X.mean(axis=1)
    MeanT0Y  = T0Y.mean(axis=1)   
    
    if DebugPlots:
        fig,ax=plt.subplots(dpi=200)
        ax.plot(T0X,T0Y,'-',lw=1,color=[0.5,0.5,0.5])
        ax.plot(MeanT0X,MeanT0Y,'-r',lw=2)
        ax.axis('equal')
        plt.show()
    
    # 3. Loop : align on temp mean contour, recompute temp mean contour, align etc...
    
    D = 5
    il = 1
    
    T0Xal = np.empty((1000,len(StackList)))
    T0Yal = np.empty((1000,len(StackList)))
    
    while D>0.1:

        for ii in range(len(StackList)):

            print('Loop ' + str(il) + ': Aligning first image on temp mean contour for ' + StackList[ii], end='\r')
            Xal,Yal = RotTransAlignement(T0X[:,ii],T0Y[:,ii],MeanT0X[:],MeanT0Y[:]
                                         ,np.linspace(-10,10,41),np.linspace(-10,10,41),np.linspace(-10,10,41),False)
            T0Xal[:,ii] = Xal
            T0Yal[:,ii] = Yal

        print('Loop ' + str(il) + ': First images aligned on temp mean contour' + '.'.ljust(20), flush=True) 

        # 4. Compute temporary mean aligned contour

        print('Loop ' + str(il) + ': Computing temp mean aligned contour')
        MeanT0Xal  = T0Xal.mean(axis=1)
        MeanT0Yal  = T0Yal.mean(axis=1)   

        if DebugPlots:
            fig,ax=plt.subplots(dpi=200)
            ax.plot(T0X,T0Y,'-',lw=2,color=[0.5,0.5,0.5])
            ax.plot(T0Xal,T0Yal,'c-',lw=1)
            ax.axis('equal')

            fig,ax=plt.subplots(dpi=200)
            ax.plot(MeanT0X,MeanT0Y,'-r',lw=3)
            ax.plot(MeanT0Xal,MeanT0Yal,'-b',lw=1)
            ax.axis('equal')
            
            plt.show()

        D = vf.dist(MeanT0X,MeanT0Y,MeanT0Xal,MeanT0Yal).mean()
        
        T0X = T0Xal[:]
        T0Y = T0Yal[:]
        MeanT0X = MeanT0Xal[:]
        MeanT0Y = MeanT0Yal[:]

        print('Loop ' + str(il) + ': Contour distance : ' + str(D))
        il = il + 1
        
         
    # Storing contour data
    for s,ii in zip(StackList,range(len(StackList))):
        data = {'Img': i*np.ones(len(T0X[:,ii])), 
                    'X': T0X[:,ii],  
                    'Y': T0Y[:,ii],
                    'S': CD.loc[(CD.index == s) & (CD['Img'] == i), 'Sparam'].values,
                    } 

        newCD = newCD.append(pd.DataFrame(data=data,index = np.repeat(s,len(T0X[:,ii]))))
  
    print('Alignement of first images done.')
    ## Alignment in time of PPGs at t = n on PPG at t = n-1
    for s,ii in zip(StackList,range(len(StackList))):       

        n = int(1 + np.max(CD.loc[s, 'Img']))
        
        print('\nAlignement in time for ' + s + ':')
        
        for i in range(1,n):
            
            print('Image ' + str(i) + '/' + str(n), end='\r')
            
            # retrieve important points and contours
            Xc = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xcenter'].values
            Yc = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Ycenter'].values

            Xparam = CD.loc[(CD.index == s) & (CD['Img'] == i), 'Xparam'].values-Xc
            Yparam = CD.loc[(CD.index == s) & (CD['Img'] == i), 'Yparam'].values-Yc

            XA = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xattach'].values-Xc
            YA = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Yattach'].values-Yc

            Xn1 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xnotch1'].values-Xc
            Yn1 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Ynotch1'].values-Yc

            Xn2 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xnotch2'].values-Xc
            Yn2 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Ynotch2'].values-Yc
            
            ## align based on the notches position

            A,R = vf.ToCirc([XA,Xn1,Xn2],[YA,Yn1,Yn2],angle='deg')
            Theta,Radius = vf.ToCirc(Xparam,Yparam,angle='deg')

            Anotches = np.mod(A[1:3].T-np.array([0,180]),360)

            Abis = np.mod(Anotches.sum()/2,360)-180

            [XAal,Xn1al,Xn2al],[YAal,Yn1al,Yn2al] = vf.ToCart(A-Abis,R,angle='deg')

            if YAal > 0:
                Abis = Abis + 180

            XparamAlnotch,YparamAlnotch = vf.ToCart(Theta-Abis,Radius,angle='deg')
            
            # retrieve contour from previous image
            XOld = newCD.loc[(newCD.index == s) & (newCD['Img'] == i-1), 'X'].values
            YOld = newCD.loc[(newCD.index == s) & (newCD['Img'] == i-1), 'Y'].values
            
            # align with previous contour
            Xal,Yal = RotTransAlignement(XparamAlnotch,YparamAlnotch,XOld[:],YOld[:]
                                         ,np.linspace(-10,10,41),np.linspace(-10,10,41),np.linspace(-10,10,41),False)
            
            # store aligned contour
            data = {'Img': i*np.ones(len(Xal)), 
                        'X': Xal,  
                        'Y': Yal,
                        'S': CD.loc[(CD.index == s) & (CD['Img'] == i), 'Sparam'].values,
                        } 

            newCD = newCD.append(pd.DataFrame(data=data,index = np.repeat(s,len(Xal))))
            
            if SavedPlots|DebugPlots:
                fig,[ax0,ax1] = plt.subplots(ncols=2,dpi=200,facecolor='black')
                fig.suptitle(s)
                ax0.plot(Xparam,Yparam,'w')
                ax0.plot(Xparam[0:-1:10],Yparam[0:-1:10],'o',color=[0.8, 0.8, 0.8],ms=3)
                ax0.plot(XA,YA,'go')
                ax0.plot(Xn1,Yn1,'c*',ms=5)
                ax0.plot(Xn2,Yn2,'m*',ms=5)
                ax0.plot(0,0,'w*')
                #ax0.set_xlim(left=-250,right=250)
                #ax0.set_ylim(top=250,bottom=-250)
                ax0.set_aspect('equal', adjustable='box')

                ax1.plot(T0X[:,ii],T0Y[:,ii],'w',lw=2,label='t0')
                ax1.plot(XOld,YOld,'b--',lw=2,label='tn-1')
                ax1.plot(Xal,Yal,'c--',lw=1,label='tn')
                #ax1.set_xlim(left=-250,right=250)
                #ax1.set_ylim(top=250,bottom=-250)
                ax1.set_aspect('equal', adjustable='box')
                plt.legend()

                fig.tight_layout()
                fig.savefig(P + '\\Figures\\Rotation\\'+ s + '\\' + str(i) +'.png') 
                if DebugPlots:
                    plt.show()
                else:
                    plt.close()
            
        print('All images aligned.\n')
                
                
    return(newCD,GD)


#%% RotTransAlignement

def RotTransAlignement(X,Y,Xref,Yref,RotRange,Xrange,Yrange,debug):

    ## Vectors sizes
    XYsize = np.size(X)
    
    if not np.size(Y)==XYsize:
        raise ValueError('X and Y must have the same size !')
    elif not np.size(Xref)==XYsize:
        raise ValueError('X and Xref must have the same size !')
    elif not np.size(Yref)==XYsize:
        raise ValueError('X and Yref must have the same size !')
    
    RotSize = np.size(RotRange)
    TXsize = np.size(Xrange)
    TYsize = np.size(Yrange)
    
    ## Average positions
    Xcenter = X.mean()
    Ycenter = Y.mean()
#     XrefCenter = Xref.mean()
#     YrefCenter = Yref.mean()
    
    ## redifine shape for 4D matrix creation
    Xref.shape=(XYsize,1,1,1) # Original coordinates, rotation, Xtranslation, Ytranslation
    Yref.shape=(XYsize,1,1,1)

    X.shape=(XYsize,1,1,1)
    Y.shape=(XYsize,1,1,1)

    RotRange.shape = (1,RotSize,1,1)
    Xrange.shape = (1,1,TXsize,1)
    Yrange.shape = (1,1,1,TYsize)

    ## Addind rotation range
    T,R = vf.ToCirc(X-Xcenter,Y-Ycenter,angle='deg')
    
    Trange = np.mod(T + RotRange,360)
    Rrange = np.tile(R,(1,RotSize,1,1))

    XrotMat,YrotMat = vf.ToCart(Trange,Rrange,angle='deg')

    XrotMat = XrotMat + Xcenter # size = (XYsize,RotSize,1,1)
    YrotMat = YrotMat + Ycenter 
    
    ## Adding X translation
    
    XrotTX = XrotMat + Xrange # size = (XYsize,RotSize,TXsize,1)
    YrotTX = np.tile(YrotMat,(1,1,TXsize,1))
    
    del(XrotMat,YrotMat)
    
    ## Adding Y translation
    
    XrotTXY = np.tile(XrotTX,(1,1,1,TYsize)) # size = (XYsize,RotSize,TXsize,TYsize)
    YrotTXY = YrotTX + Yrange
    
    del(XrotTX,YrotTX)
    
    ## Duplicating references 

    Xref = np.tile(Xref,(1,RotSize,TXsize,TYsize))
    Yref = np.tile(Yref,(1,RotSize,TXsize,TYsize))
    
    
    ## Mean square istance computation

    D = vf.dist(XrotTXY,YrotTXY,Xref,Yref)
    

    SD = np.square(D)

    MSD = np.mean(SD,axis=0)
    
    ## finding best alignement
    
    ind = np.unravel_index(np.argmin(MSD), MSD.shape)

    Xal = XrotTXY[:,ind[0],ind[1],ind[2]]
    Yal = YrotTXY[:,ind[0],ind[1],ind[2]]

#     fig, ax = plt.subplots(dpi=150)
#     fig.suptitle('X and Y')
#     sns.heatmap(MSD[],cmap = 'gist_rainbow',vmin=0)

    if debug:
        ## Result plotting
        f,ax = plt.subplots(dpi=200,figsize=(5,5))
        f.suptitle('Contour alignement result')
        ax.plot(X[:,0,0,0],Y[:,0,0,0],'-b',label='Contour to align')
        ax.plot(Xref[:,0,0,0],Yref[:,0,0,0],'-w',label='Reference contour')
        ax.plot(Xal,Yal,'m-',ms=2,lw=1,label='Aligned contour')
        ax.set_aspect('equal',adjustable='box')
        plt.legend(fontsize = 'xx-small')
        plt.show()

        ## Variable space visualisation

        plt.rcParams.update({'font.size': 6})

        f,[ax1,ax2] = plt.subplots(dpi=200,ncols=2)

        ax1.set_title('at original rotation')
        ax1.set_aspect(TXsize/TYsize)
        sns.heatmap(MSD[int(np.round(RotSize/2)),:,:],cmap = 'gist_rainbow',
                         vmin=0,ax=ax1,cbar_kws = {'shrink':0.5})
        ax1.set_xlabel('Y translation')
        ax1.set_ylabel('X translation')
        ax1.set_xticks(np.linspace(0,TYsize,10))
        ax1.set_xticklabels(Yrange[0,0,0,0:-1:int(np.round(TYsize/10))])
        ax1.set_yticks(np.linspace(0,TXsize,10))
        ax1.set_yticklabels(Xrange[0,0,0:-1:int(np.round(TXsize/10)),0])

        ax2.set_title('at aligned rotation')
        ax2.set_aspect(TXsize/TYsize)
        sns.heatmap(MSD[ind[0],:,:],cmap = 'gist_rainbow',
                         vmin=0,ax=ax2,cbar_kws = {'shrink':0.5})
        ax2.set_xlabel('Y translation')
        ax2.set_ylabel('X translation')
        ax2.set_xticks(np.linspace(0,TYsize,10))
        ax2.set_xticklabels(Yrange[0,0,0,0:-1:int(np.round(TYsize/10))])
        ax2.set_yticks(np.linspace(0,TXsize,10))
        ax2.set_yticklabels(Xrange[0,0,0:-1:int(np.round(TXsize/10)),0])

        f.tight_layout()


        f,[ax1,ax2] = plt.subplots(dpi=200,ncols=2)

        ax1.set_title('at original Y')
        ax1.set_aspect(TXsize/RotSize)
        sns.heatmap(MSD[:,:,int(np.round(TYsize/2))],cmap = 'gist_rainbow',
                         vmin=0,ax=ax1,cbar_kws = {'shrink':0.5})
        ax1.set_xlabel('X translation')
        ax1.set_ylabel('Rotation')
        ax1.set_xticks(np.linspace(0,TXsize,10))
        ax1.set_xticklabels(Xrange[0,0,0:-1:int(np.round(TXsize/10)),0])
        ax1.set_yticks(np.linspace(0,RotSize,10))
        ax1.set_yticklabels(RotRange[0,0:-1:int(np.round(RotSize/10)),0,0])

        ax2.set_title('at aligned Y')
        ax2.set_aspect(TXsize/RotSize)
        sns.heatmap(MSD[:,:,ind[2]],cmap = 'gist_rainbow',
                         vmin=0,ax=ax2,cbar_kws = {'shrink':0.5})
        ax2.set_xlabel('X translation')
        ax2.set_ylabel('Rotation')
        ax2.set_xticks(np.linspace(0,TXsize,10))
        ax2.set_xticklabels(Xrange[0,0,0:-1:int(np.round(TXsize/10)),0])
        ax2.set_yticks(np.linspace(0,RotSize,10))
        ax2.set_yticklabels(RotRange[0,0:-1:int(np.round(RotSize/10)),0,0])

        f.tight_layout()


        f,[ax1,ax2] = plt.subplots(dpi=200,ncols=2)

        ax1.set_title('at original X')
        ax1.set_aspect(TYsize/RotSize)
        sns.heatmap(MSD[:,int(np.round(TXsize/2)),:],cmap = 'gist_rainbow',
                         vmin=0,ax=ax1,cbar_kws = {'shrink':0.5})
        ax1.set_xlabel('Y translation')
        ax1.set_ylabel('Rotation')
        ax1.set_xticks(np.linspace(0,TYsize,10))
        ax1.set_xticklabels(Yrange[0,0,0,0:-1:int(np.round(TYsize/10))])
        ax1.set_yticks(np.linspace(0,RotSize,10))
        ax1.set_yticklabels(RotRange[0,0:-1:int(np.round(RotSize/10)),0,0])

        ax2.set_title('at aligned X')
        ax2.set_aspect(TYsize/RotSize)
        sns.heatmap(MSD[:,ind[1],:],cmap = 'gist_rainbow',
                         vmin=0,ax=ax2,cbar_kws = {'shrink':0.5})
        ax2.set_xlabel('Y translation')
        ax2.set_ylabel('Rotation')
        ax2.set_xticks(np.linspace(0,TYsize,10))
        ax2.set_xticklabels(Yrange[0,0,0,0:-1:int(np.round(TYsize/10))])
        ax2.set_yticks(np.linspace(0,RotSize,10))
        ax2.set_yticklabels(RotRange[0,0:-1:int(np.round(RotSize/10)),0,0])

        f.tight_layout()
    
    del(XrotTXY,YrotTXY,RotRange,Xrange,Yrange,X,Y,Xref,Yref)
    
    gc.collect();
    
    return(Xal,Yal)
    
#%% curvAbsci

def curvAbsci(CD,GD,StackList,P, **kwargs):

    if not os.path.exists(P + '\\Figures\\Parametrisation\\'):
        os.mkdir(P + '\\Figures\\Parametrisation\\') # create folder
    
    DebugPlots = False
    SavedPlots = False
    
    for key, value in kwargs.items(): 
        if key == 'debug':
            DebugPlots = value
        if key == 'saveplots':
            SavedPlots = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    newCD = pd.DataFrame(data=None,columns=['Img','Xparam','Yparam','Sparam']) 
    
    
    for s in StackList:
        
        print('Processing ' + s + '...',end=' ')
        
        if not os.path.exists(P + '\\Figures\\Parametrisation\\'+ s + '\\'):
            os.mkdir(P + '\\Figures\\Parametrisation\\'+ s + '\\') # create folder
            
        n = int(1 + np.max(CD.loc[s, 'Img']))
        
        for i in range(n): # 
            
            # retrieve important points and contours
            Xcont = CD.loc[(CD.index == s) & (CD['Img'] == i), 'Xsmooth'].values
            Ycont = CD.loc[(CD.index == s) & (CD['Img'] == i), 'Ysmooth'].values            
            
            Xc = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xcenter'].values
            Yc = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Ycenter'].values
            
            Xat = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xattach'].values
            Yat = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Yattach'].values

            Xn1 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xnotch1'].values
            Yn1 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Ynotch1'].values

            Xn2 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xnotch2'].values
            Yn2 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Ynotch2'].values
            
            locn1 = np.argmin(np.abs(np.sqrt(np.square(Xcont-Xn1)+np.square(Ycont-Yn1))))
            locn2 = np.argmin(np.abs(np.sqrt(np.square(Xcont-Xn2)+np.square(Ycont-Yn2))))
            
            
            # identify 'first notch' as the one being after the attach point clockwise
            # Shift the contour points so that they start by the first notch
            Aat,Rat = vf.ToCirc(Xat-Xc,Yat-Yc,angle='deg')
            An1,Rn1 = vf.ToCirc(Xn1-Xc,Yn1-Yc,angle='deg')
            An2,Rn2 = vf.ToCirc(Xn2-Xc,Yn2-Yc,angle='deg')
            
            An1=np.mod(An1-Aat,360)
            An2=np.mod(An2-Aat,360)
            
            if An1>An2: # n1 is first notch
                Xcont = np.roll(Xcont,-locn1)
                Ycont = np.roll(Ycont,-locn1)
                locn = locn2-locn1 # non starting notch
                GD.loc[(GD.index == s) & (GD['Img'] == i) ,'LocNotch1Param'] = 0
                GD.loc[(GD.index == s) & (GD['Img'] == i),'LocNotch2Param'] = locn
                    
            else: # n2 is first notch
                Xcont = np.roll(Xcont,-locn2)
                Ycont = np.roll(Ycont,-locn2)
                locn = locn1-locn2 # non starting notch
                GD.loc[(GD.index == s) & (GD['Img'] == i) ,'LocNotch1Param'] = locn
                GD.loc[(GD.index == s) & (GD['Img'] == i),'LocNotch2Param'] = 0
            

            # Computes length along contour and total length
            ContourCumLength = np.concatenate(([0],np.cumsum(np.sqrt(np.square(np.diff(Xcont))+np.square(np.diff(Ycont))))))
            ContourLength = ContourCumLength[-1]
            
            # Create contour interpolation function based on length along contour
            ContourInterp = interp1d(ContourCumLength,[Xcont,Ycont], fill_value='extrapolate')            
            
            ## Create parametrisation for two section of contour : upper section between the two notches not containing the attach 
            ## point & lower section between the two noteches containing the attach points
            
            npts = 500 # number of point per section of contour
            
            # Upper section
            SegmentLength_up = ContourCumLength[locn]
            deltaL_up = SegmentLength_up/npts
            SegmentRegCumLength_up = np.linspace(0,npts,npts+1)*deltaL_up

            Xparam_up,Yparam_up = ContourInterp(SegmentRegCumLength_up[0:-1])
            Sparam_up = (SegmentRegCumLength_up[0:-1])/ContourLength
            
            CurvContourLength_up = np.sum(np.sqrt(np.square(np.diff(Xparam_up))+np.square(np.diff(Yparam_up))))
            
            # Lower section
            SegmentLength_lo = ContourLength-ContourCumLength[locn]
            deltaL_lo = SegmentLength_lo/npts 
            SegmentRegCumLength_lo = np.linspace(0,npts,npts+1)*deltaL_lo+SegmentLength_up

            Xparam_lo,Yparam_lo = ContourInterp(SegmentRegCumLength_lo[0:-1])
            Sparam_lo = (SegmentRegCumLength_lo[0:-1])/ContourLength
            
            CurvContourLength_lo = np.sum(np.sqrt(np.square(np.diff(Xparam_lo))+np.square(np.diff(Yparam_lo))))
            
            
            if DebugPlots:
                print(s + '_' + str(i) + ' contour length variation : ' + 
                      str(round((ContourLength-(CurvContourLength_up+CurvContourLength_lo+deltaL_up+deltaL_lo))
                                /ContourLength*10000)/100) + '%')

                
            Xparam = np.concatenate((Xparam_up, Xparam_lo))
            Yparam = np.concatenate((Yparam_up, Yparam_lo))
            Sparam = np.concatenate((Sparam_up, Sparam_lo))
            
            if SavedPlots|DebugPlots:
                
                loca = np.argmin(np.abs(np.sqrt(np.square(Xparam-Xat)+np.square(Yparam-Yat))))
                
                fig1, ax1 = plt.subplots(dpi = 200,facecolor='black')
                fig1.suptitle(s)
                ax1.set_title("Regular parametrisation by segment (1/10 pts)")
                ax1.plot(Xparam-Xc,Yparam-Yc,'w-')
                ax1.plot(0,0,'w*')
                ax1.plot(Xparam[0]-Xc,Yparam[0]-Yc,'mo',ms = 5, label='First notch')
                ax1.plot(Xcont[locn]-Xc,Ycont[locn]-Yc,'co',ms = 5, label='Second notch')
                ax1.plot(Xparam[loca]-Xc,Yparam[loca]-Yc,'go',ms = 5, label='Attach')
                ax1.plot(Xparam_up[0:len(Xparam_up)-1:10]-Xc,Yparam_up[0:len(Xparam_up)-1:10]-Yc,'bo',ms = 3,label='Upper section')
                ax1.plot(Xparam_lo[0:len(Xparam_lo)-1:10]-Xc,Yparam_lo[0:len(Xparam_lo)-1:10]-Yc,'ro',ms = 3, label='Lower section')
                ax1.set_aspect('equal', adjustable='box')
                plt.legend(fontsize = 'xx-small')
                fig1.savefig(P + '\\Figures\\Parametrisation\\'+ s + '\\' + 'Reg_' + str(i) +'.png') 
                if DebugPlots & ((i == 0)|(i == 20)|(i == 30)|(i == 10)):
                    plt.show()
                else:
                    plt.close()
                
            # Contour normalisation
            Thp,Rp = vf.ToCirc(Xparam,Yparam)
            XcN,YcN = vf.ToCart(Thp,np.divide(Rp,np.median(Rp)))
            
            # Storing contour data
            data = {'Img': i*np.ones(len(Xparam)),
                        'Xparam': Xparam,
                        'Yparam': Yparam,
                        'XparamNorm': XcN,
                        'YparamNorm': YcN,
                        'Sparam': Sparam} 
            

            
            newCD = newCD.append(pd.DataFrame(data=data,index = np.repeat(s,len(Xparam))) )            
        print('Done')
            
    return(newCD,GD)

#%% getLandmarks

def getLandmarks(CD,GD,StackList,Scale,P,stringName, **kwargs): 
        
    # Utility function
    def round_to_odd(f):
        return np.floor(f) // 2 * 2 + 1    
    
    # Folder definition
    if not os.path.exists(P + '\\Figures\\'):
        os.mkdir(P + '\\Figures\\') # create folder
        
    if not os.path.exists(P + '\\Figures\\Landmarks\\'):
        os.mkdir(P + '\\Figures\\Landmarks\\') # create folder
        
    # Kwargs
    DebugPlots = False
    SavedPlots = False
    Dmax = 20
    Dmax2 = 30
    AUTO = False
    OverwriteData = False
    
    for key, value in kwargs.items(): 
        if key == 'debug':
            DebugPlots = value
        elif key == 'saveplots':
            SavedPlots = value
        elif key == 'Dmax':
            Dmax = value
        elif key == 'Dmax2':
            Dmax2 = value
        elif key == 'Auto':
            AUTO = value
        elif key == 'Overwrite':
            OverwriteData = value             
        elif key == 'FirstSlice':
            FirstSlice = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
        
    Dmax = Dmax/Scale
    Dmax2 = Dmax2/Scale
    
    # Load or create file for landmark points 
    if os.path.exists(P + '\\clickedpoints.csv'):
        RefPts = pd.read_csv(P + '\\clickedpoints.csv', index_col = 'Ind')
    else:                
        RefPts = pd.DataFrame(data=None,columns=['Img','Xnotch1','Ynotch1','Xnotch2','Ynotch2','Xattach','Yattach']) 
    
    # 1. For each video in StackList identify landmarks on first image
    
    for s in StackList:
        
        if not os.path.exists(P + '\\Figures\\Landmarks\\'+ s + '\\'):
            os.mkdir(P + '\\Figures\\Landmarks\\'+ s + '\\') # create folder
            
        print('First image landmarks for : ' + s.ljust(10), flush=True, end = '\r')
        
        i = 0 # First image
        
        if ('LocNotch2' in GD) & (not OverwriteData):
            if (np.isnan(GD.loc[(GD.index == s) & (GD['Img'] == i) ,'LocNotch2'].values[:])):
                DO = True
            else:
                DO = False
        else:
            DO = True
        
        if DO:
        
            # Loading contour and center coordinate
            Xcenter = GD.loc[(GD.index == s) & (GD['Img'] == i) ,'Xcenter'].values
            Ycenter = GD.loc[(GD.index == s) & (GD['Img'] == i) ,'Ycenter'].values

            Xcont = CD.loc[(CD.index == s) & (CD['Img'] == i) ,'Ximg'].values
            Ycont = CD.loc[(CD.index == s) & (CD['Img'] == i) ,'Yimg'].values


            
            # Contour smoothing
            window = int(round_to_odd(len(Xcont)/50))

            Xsmooth = savgol_filter(Xcont, window, 3)
            Ysmooth = savgol_filter(Ycont, window, 3)

            # Save smoothed contour
            CD.loc[(CD.index == s) & (CD['Img'] == i) ,'Xsmooth'] = Xsmooth
            CD.loc[(CD.index == s) & (CD['Img'] == i) ,'Ysmooth'] = Ysmooth

            ### Automaticaly detect points of high inward curvature  
            # Compute curvature
            Step = round(2/100*len(Xsmooth))
            xc,yc,Curv = getContourCurvature(Xsmooth,Ysmooth,Step)

            # Compute signed curvature using the position of the center of the fitted 
            # circle with regards to the contour
            poly = [(x,y) for (x,y) in zip(Xsmooth,Ysmooth)]
            points = [[x,y] for (x,y) in zip(xc,yc)]
            path = mpltPath.Path(poly)            
            isIn = path.contains_points(points)

            CurvSign = []
            for Bool in isIn:
                if Bool:
                    CurvSign.append(1)
                else:
                    CurvSign.append(-1)

            SignedCurv = np.multiply(Curv,CurvSign)

            # Detecting negative peaks in signed curvature
            loc,prop = find_peaks(-SignedCurv, distance = len(SignedCurv)/50) 


            ### Identifying landmarks
            # if clicked points not already in the file, ask user and then save in the file
            if RefPts.loc[(RefPts.index == s)&(RefPts['Img']==i)].empty:

                # identifying notches manually on fist image
                NotchesRef = getContourPointsCoordinates(P,s,2,i,Xsmooth,Ysmooth,Xsmooth[loc],Ysmooth[loc],Xsmooth,Ysmooth,Xsmooth[loc],Ysmooth[loc],'Select the two notches')
                AttachRef = getContourPointsCoordinates(P,s,1,i,Xsmooth,Ysmooth,Xsmooth[loc],Ysmooth[loc],Xsmooth,Ysmooth,Xsmooth[loc],Ysmooth[loc],'Select the attach point')

                # Save the points that were clicked
                data = {'Img': i,
                        'Xnotch1':NotchesRef[0][0],
                        'Ynotch1':NotchesRef[0][1],
                        'Xnotch2':NotchesRef[1][0],
                        'Ynotch2':NotchesRef[1][1],
                        'Xattach':AttachRef[0][0],
                        'Yattach':AttachRef[0][1]}            

                RefPts = RefPts.append(pd.DataFrame(data=data,index = [s]))

                RefPts.to_csv(P + '\\clickedpoints.csv',index_label = 'Ind')

            else: # Load clicked points from the file
                    NotchesRef = [[RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i) ,'Xnotch1'].values,
                                       RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i) ,'Ynotch1'].values],
                                     [RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i) ,'Xnotch2'].values,
                                       RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i) ,'Ynotch2'].values]]
                    AttachRef = [[RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i) ,'Xattach'].values, 
                                 RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i) ,'Yattach'].values]]


            # Find landmarks in contour from clicked points and Curvature computation   
            xN1 = Xsmooth[loc][np.argmin(np.sqrt(np.square(Xsmooth[loc]-NotchesRef[0][0])+
                                                  np.square(Ysmooth[loc]-NotchesRef[0][1])))]  
            yN1 = Ysmooth[loc][np.argmin(np.sqrt(np.square(Xsmooth[loc]-NotchesRef[0][0])+
                                                  np.square(Ysmooth[loc]-NotchesRef[0][1])))] 

            xN2 = Xsmooth[loc][np.argmin(np.sqrt(np.square(Xsmooth[loc]-NotchesRef[1][0])+
                                                  np.square(Ysmooth[loc]-NotchesRef[1][1])))]  
            yN2 = Ysmooth[loc][np.argmin(np.sqrt(np.square(Xsmooth[loc]-NotchesRef[1][0])+
                                                  np.square(Ysmooth[loc]-NotchesRef[1][1])))] 

            xA = Xsmooth[loc][np.argmin(np.sqrt(np.square(Xsmooth[loc]-AttachRef[0][0])+
                                                  np.square(Ysmooth[loc]-AttachRef[0][1])))]  
            yA = Ysmooth[loc][np.argmin(np.sqrt(np.square(Xsmooth[loc]-AttachRef[0][0])+
                                                  np.square(Ysmooth[loc]-AttachRef[0][1])))] 

            # Save landmarks points
            GD.loc[(GD.index == s) & (GD['Img'] == i) ,'Xattach'] = xA
            GD.loc[(GD.index == s) & (GD['Img'] == i) ,'Yattach'] = yA

            GD.loc[(GD.index == s) & (GD['Img'] == i) ,'Xnotch1'] = xN1
            GD.loc[(GD.index == s) & (GD['Img'] == i) ,'Ynotch1'] = yN1

            GD.loc[(GD.index == s) & (GD['Img'] == i),'Xnotch2'] = xN2
            GD.loc[(GD.index == s) & (GD['Img'] == i),'Ynotch2'] = yN2


            # Find and save the positions in contour of notches points
            locn1 = np.argmin(np.abs(np.sqrt(np.square(Xsmooth-xN1)+np.square(Ysmooth-yN1))))
            locn2 = np.argmin(np.abs(np.sqrt(np.square(Xsmooth-xN2)+np.square(Ysmooth-yN2))))

            GD.loc[(GD.index == s) & (GD['Img'] == i) ,'LocNotch1'] = locn1
            GD.loc[(GD.index == s) & (GD['Img'] == i) ,'LocNotch2'] = locn2

            # Plotting 
            if SavedPlots:
                RGBimg = io.imread(P + '\\' + s + '.tif', key = i)

                fig0, [ax0, ax1] = plt.subplots(ncols = 2, dpi = 250,facecolor='black')
                fig0.suptitle(s)
                ax0.set_title('Attach point (magenta), notches (red),\n and center (blue) for alignement',fontsize=8)
                ax0.imshow(RGBimg)
                ax0.plot(Xsmooth,Ysmooth,'c',lw = 0.5)
                ax0.plot(xN1,yN1,'.r',ms=2)
                ax0.plot(xN2,yN2,'.r',ms=2)
                ax0.plot(xA,yA,'.m',ms=2)
                ax0.plot(Xcenter[0],Ycenter[0],'*b', ms = 3)
                ax0.set_xticks([], [])
                ax0.set_yticks([], [])
                sc = ax1.scatter(Xsmooth,-Ysmooth+2*Ycenter[0], c= SignedCurv, cmap = 'BrBG', s = 2)
                ax1.plot(xN1,-yN1+2*Ycenter[0],'ob',ms=3)
                ax1.plot(xN2,-yN2+2*Ycenter[0],'ob',ms=3)
                ax1.plot(xA,-yA+2*Ycenter[0],'sb',ms=3)
                ax1.plot(Xsmooth[loc],-Ysmooth[loc]+2*Ycenter[0],'r+',mfc='none', ms = 3, lw= 0.5)
                ax1.set_aspect('equal')
                ax1.set_xticks([], [])
                ax1.set_yticks([], [])
                #ax1.set_xlim(left=-250+Xcenter[0],right=250+Xcenter[0])
                #ax1.set_ylim(top=250+Ycenter[0],bottom=-250+Ycenter[0])
                fig0.colorbar(sc, ax = ax1, label = 'Curvature',shrink = 0.6)
                fig0.tight_layout()

                fig0.savefig(P + '\\Figures\\Landmarks\\'+ s + '\\' + str(i) +'.png') 
                plt.close()

    # Save tmp file with already analyzed stacks        
    GD.to_csv(P + '\\GlobalData' + stringName + '_Landmarks_tmp.csv',index_label = 'Ind')
    CD.to_csv(P + '\\ContourData' + stringName + '_Landmarks_tmp.csv',index_label = 'Ind')
    
    print('\n')
    
    if not FirstSlice:
        # 2. For the remaining image, detect automatically landmark points
        for s in StackList:       

            nimg = int(1 + np.max(GD.loc[s, 'Img'])) # number of images in the stack

            if  OverwriteData:
                idx = 1
            else:            
                List = GD.loc[s,'LocNotch2'].values
                idx = next((i for i in range(len(List)) if np.isnan(List[i])),nimg)
    
#             print('nimg : ' + str(nimg) + ' // idx : ' + str(idx))

            if idx == nimg:
                print('Stack ' + s + ' already processed in tmp file.', end='')
            else:
                for i in range(idx,nimg):

                    print('Processing ' + s + ' image ' + str(i+1) + '/' + str(nimg).ljust(10), flush=True, end = '\r')

                    # Loading contour and center coordinate
                    Xcenter = GD.loc[(GD.index == s) & (GD['Img'] == i) ,'Xcenter'].values
                    Ycenter = GD.loc[(GD.index == s) & (GD['Img'] == i) ,'Ycenter'].values

                    Xcont = CD.loc[(CD.index == s) & (CD['Img'] == i) ,'Ximg'].values
                    Ycont = CD.loc[(CD.index == s) & (CD['Img'] == i) ,'Yimg'].values
                    
                    # Contour smoothing
                    window = int(round_to_odd(len(Xcont)/50))

                    Xsmooth = savgol_filter(Xcont, window, 3)
                    Ysmooth = savgol_filter(Ycont, window, 3)

                    # Save smoothed contour
                    CD.loc[(CD.index == s) & (CD['Img'] == i) ,'Xsmooth'] = Xsmooth
                    CD.loc[(CD.index == s) & (CD['Img'] == i) ,'Ysmooth'] = Ysmooth

                    # Center smoothed contour on (0,0)
                    Xcent = Xsmooth - Xcenter
                    Ycent = Ysmooth - Ycenter

                    # Getting centered contour from previous image 
                    XsmoothOld = CD.loc[(CD.index == s) & (CD['Img'] == i-1) ,'Xsmooth'].values
                    YsmoothOld = CD.loc[(CD.index == s) & (CD['Img'] == i-1) ,'Ysmooth'].values

                    XcenterOld = GD.loc[(GD.index == s) & (GD['Img'] == i-1) ,'Xcenter'].values
                    YcenterOld = GD.loc[(GD.index == s) & (GD['Img'] == i-1) ,'Ycenter'].values

                    XcentOld = XsmoothOld-XcenterOld
                    YcentOld = YsmoothOld-YcenterOld

                    ### Rotate new contour to align on old one
                    # Expressed in polar coordinates
                    Theta,R = vf.ToCirc(XcentOld,YcentOld,angle = 'deg')
                    Rs = mtl.repmat(R,360,1)

                    # Rotation steps (1°)
                    rot = np.transpose(np.array([range(0,360)]))
                    Thetas = np.mod(Theta + rot,360)

                    Xmatrot, Ymatrot = vf.ToCart(Thetas,Rs, angle = 'deg')

                    Dvect = np.empty(360)

                    for d in range(360):
                        Nxy = len(Xcent)
                        minD = np.empty(Nxy)
                        for (x,y,nxy) in zip(Xcent,Ycent,range(Nxy)):
                             minD[nxy] = np.min(vf.dist(x,y,Xmatrot[d,:],Ymatrot[d,:]))
                        Dvect[d] = sum(np.square(minD))

                    m = np.argmin(Dvect)

                    XalOld = Xmatrot[m,:]+Xcenter
                    YalOld = Ymatrot[m,:]+Ycenter

                    if DebugPlots:
                        mp = np.mod(m+5,359)
                        mm = np.mod(m-5,359)
                        fig,ax = plt.subplots(dpi=200)
                        ax.set_title('Contour alignement for image : ' +str(i) + ' m = ' + str(m) )
                        ax.set_aspect('equal')
                        ax.plot(XsmoothOld,YsmoothOld,'b--',label = ('i-1 contour'))
                        ax.plot(Xsmooth,Ysmooth,'w--', label = ('i contour'))
                        ax.plot(XalOld,YalOld,'g',label = ('translated and rotated i-1 contour'))
                        ax.plot(Xmatrot[mp,:]+Xcenter,Ymatrot[mp,:]+Ycenter,'r',lw=1,label = ('translated and rotated i-1 contour'))
                        ax.plot(Xmatrot[mm,:]+Xcenter,Ymatrot[mm,:]+Ycenter,'r--',lw=1,label = ('translated and rotated i-1 contour'))
                        plt.legend(fontsize = 'xx-small')
                        plt.show()
                    
                    
                    ### Automaticaly detect points of high inward curvature in new contour
                    # Compute curvature
                    Step = round(2/100*len(Xsmooth))
                    xc,yc,Curv = getContourCurvature(Xsmooth,Ysmooth,Step)

                    # Compute signed curvature using the position of the center of the fitted 
                    # circle with regards to the contour
                    poly = [(x,y) for (x,y) in zip(Xsmooth,Ysmooth)]
                    points = [[x,y] for (x,y) in zip(xc,yc)]
                    path = mpltPath.Path(poly)            
                    isIn = path.contains_points(points)

                    CurvSign = []
                    for Bool in isIn:
                        if Bool:
                            CurvSign.append(1)
                        else:
                            CurvSign.append(-1)

                    SignedCurv = np.multiply(Curv,CurvSign)

                    # Detecting peaks in signed curvature
                    loc,prop = find_peaks(-SignedCurv, distance = len(SignedCurv)/30)             

                    ### Identifying landmarks
                    # if there is no reference point in the file
                    if RefPts.loc[(RefPts.index == s)&(RefPts['Img']==i)].empty:
                        
                        ### if high curvature points are too far from old 
                        ### aligned contour landmarks, ask again the user

                        # landmark points on old contour, centered
                        xN1Old = GD.loc[(GD.index == s) & (GD['Img'] == i-1) ,'Xnotch1'].values-XcenterOld
                        yN1Old = GD.loc[(GD.index == s) & (GD['Img'] == i-1) ,'Ynotch1'].values-YcenterOld
                        xN2Old = GD.loc[(GD.index == s) & (GD['Img'] == i-1) ,'Xnotch2'].values-XcenterOld
                        yN2Old = GD.loc[(GD.index == s) & (GD['Img'] == i-1) ,'Ynotch2'].values-YcenterOld
                        xAOld = GD.loc[(GD.index == s) & (GD['Img'] == i-1) ,'Xattach'].values-XcenterOld
                        yAOld = GD.loc[(GD.index == s) & (GD['Img'] == i-1) ,'Yattach'].values-YcenterOld


                        # rotation of old landmark points
                        ThetaN1Old,RN1Old = vf.ToCirc(xN1Old,yN1Old, angle = 'deg')
                        xN1OldRot,yN1OldRot = vf.ToCart(ThetaN1Old+m,RN1Old, angle = 'deg')
                        ThetaN2Old,RN2Old = vf.ToCirc(xN2Old,yN2Old, angle = 'deg')
                        xN2OldRot,yN2OldRot = vf.ToCart(ThetaN2Old+m,RN2Old, angle = 'deg')
                        ThetaAOld,RAOld = vf.ToCirc(xAOld,yAOld, angle = 'deg')
                        xAOldRot,yAOldRot = vf.ToCart(ThetaAOld+m,RAOld, angle = 'deg')

                        # Old landmark points aligned to new contour
                        xN1OldAl = xN1OldRot+Xcenter
                        yN1OldAl = yN1OldRot+Ycenter
                        xN2OldAl = xN2OldRot+Xcenter
                        yN2OldAl = yN2OldRot+Ycenter
                        xAOldAl = xAOldRot+Xcenter
                        yAOldAl = yAOldRot+Ycenter


                        DN1 = np.min(vf.dist(xN1OldAl,yN1OldAl,Xsmooth[loc],Ysmooth[loc]))
                        DN2 = np.min(vf.dist(xN2OldAl,yN2OldAl,Xsmooth[loc],Ysmooth[loc]))
                        DN = np.max([DN1,DN2])

                        DA = np.min(vf.dist(xAOldAl,yAOldAl,Xsmooth[loc],Ysmooth[loc]))

                        if (DN > Dmax):
                            if AUTO:
                                # aligned old landlark points (= new reference points)
                                NotchesRef = [[xN1OldAl,yN1OldAl],
                                             [xN2OldAl,yN2OldAl]]
                            else:
                                NotchesRef = getContourPointsCoordinates(P,s,2,i,Xsmooth,Ysmooth,Xsmooth[loc],
                                                                         Ysmooth[loc],XalOld,YalOld,[xN1OldAl,xN2OldAl],[yN1OldAl,yN2OldAl], 'Select the two notches')

                        else:
                            # aligned old landlark points (= new reference points)
                            NotchesRef = [[xN1OldAl,yN1OldAl],
                                         [xN2OldAl,yN2OldAl]]

                        if (DA > Dmax):

                            if AUTO:
                               # aligned old landlark points (= new reference points)
                                AttachRef = [[xAOldAl,yAOldAl]]
                            else:
                                AttachRef = getContourPointsCoordinates(P,s,1,i,Xsmooth,Ysmooth,Xsmooth[loc],
                                                                        Ysmooth[loc],XalOld,YalOld,xAOldAl,yAOldAl,'Select the attach point')

                        else:  
                            # aligned old landlark points (= new reference points)
                            AttachRef = [[xAOldAl,yAOldAl]]



                        # Save the reference points
                        data = {'Img': i,
                                'Xnotch1':NotchesRef[0][0],
                                'Ynotch1':NotchesRef[0][1],
                                'Xnotch2':NotchesRef[1][0],
                                'Ynotch2':NotchesRef[1][1],
                                'Xattach':AttachRef[0][0],
                                'Yattach':AttachRef[0][1]}            

                        RefPts = RefPts.append(pd.DataFrame(data=data,index = [s]))

                        RefPts.to_csv(P + '\\clickedpoints.csv',index_label = 'Ind')
                        
                    else:
                        NotchesRef = [[RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i) ,'Xnotch1'].values,
                                           RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i) ,'Ynotch1'].values],
                                         [RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i) ,'Xnotch2'].values,
                                           RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i) ,'Ynotch2'].values]]
                        AttachRef = [[RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i) ,'Xattach'].values, 
                                     RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i) ,'Yattach'].values]]
                        
                        DN1 = np.min(vf.dist(NotchesRef[0][0],NotchesRef[0][1],Xsmooth[loc],Ysmooth[loc]))
                        DN2 = np.min(vf.dist(NotchesRef[1][0],NotchesRef[1][1],Xsmooth[loc],Ysmooth[loc]))
                        DN = np.max([DN1,DN2])

                        DA = np.min(vf.dist(AttachRef[0][0],AttachRef[0][1],Xsmooth[loc],Ysmooth[loc]))


                    # Getting new landmarks  
                    if (DN > Dmax2):
                        xN1 = Xsmooth[np.argmin(np.sqrt(np.square(Xsmooth-NotchesRef[0][0])+
                                                              np.square(Ysmooth-NotchesRef[0][1])))]  
                        yN1 = Ysmooth[np.argmin(np.sqrt(np.square(Xsmooth-NotchesRef[0][0])+
                                                              np.square(Ysmooth-NotchesRef[0][1])))] 

                        xN2 = Xsmooth[np.argmin(np.sqrt(np.square(Xsmooth-NotchesRef[1][0])+
                                                              np.square(Ysmooth-NotchesRef[1][1])))]  
                        yN2 = Ysmooth[np.argmin(np.sqrt(np.square(Xsmooth-NotchesRef[1][0])+
                                                              np.square(Ysmooth-NotchesRef[1][1])))] 
                    else:
                        xN1 = Xsmooth[loc][np.argmin(np.sqrt(np.square(Xsmooth[loc]-NotchesRef[0][0])+
                                                              np.square(Ysmooth[loc]-NotchesRef[0][1])))]  
                        yN1 = Ysmooth[loc][np.argmin(np.sqrt(np.square(Xsmooth[loc]-NotchesRef[0][0])+
                                                              np.square(Ysmooth[loc]-NotchesRef[0][1])))] 

                        xN2 = Xsmooth[loc][np.argmin(np.sqrt(np.square(Xsmooth[loc]-NotchesRef[1][0])+
                                                              np.square(Ysmooth[loc]-NotchesRef[1][1])))]  
                        yN2 = Ysmooth[loc][np.argmin(np.sqrt(np.square(Xsmooth[loc]-NotchesRef[1][0])+
                                                              np.square(Ysmooth[loc]-NotchesRef[1][1])))] 
                        
                    if (DA > Dmax2):
                        xA = Xsmooth[np.argmin(np.sqrt(np.square(Xsmooth-AttachRef[0][0])+
                                                              np.square(Ysmooth-AttachRef[0][1])))]  
                        yA = Ysmooth[np.argmin(np.sqrt(np.square(Xsmooth-AttachRef[0][0])+
                                                              np.square(Ysmooth-AttachRef[0][1])))] 
                    else:
                        xA = Xsmooth[loc][np.argmin(np.sqrt(np.square(Xsmooth[loc]-AttachRef[0][0])+
                                                              np.square(Ysmooth[loc]-AttachRef[0][1])))]  
                        yA = Ysmooth[loc][np.argmin(np.sqrt(np.square(Xsmooth[loc]-AttachRef[0][0])+
                                                              np.square(Ysmooth[loc]-AttachRef[0][1])))] 

                    
                    # saving
                    GD.loc[(GD.index == s) & (GD['Img'] == i) ,'Xattach'] = xA
                    GD.loc[(GD.index == s) & (GD['Img'] == i) ,'Yattach'] = yA

                    GD.loc[(GD.index == s) & (GD['Img'] == i) ,'Xnotch1'] = xN1
                    GD.loc[(GD.index == s) & (GD['Img'] == i) ,'Ynotch1'] = yN1

                    GD.loc[(GD.index == s) & (GD['Img'] == i),'Xnotch2'] = xN2
                    GD.loc[(GD.index == s) & (GD['Img'] == i),'Ynotch2'] = yN2

                    
                    # Find and save position of notches 
                    locn1 = np.argmin(np.abs(np.sqrt(np.square(Xsmooth-xN2)+np.square(Ysmooth-yN2))))
                    locn2 = np.argmin(np.abs(np.sqrt(np.square(Xsmooth-xN1)+np.square(Ysmooth-yN1))))
                    
                    GD.loc[(GD.index == s) & (GD['Img'] == i) ,'LocNotch1'] = locn1
                    GD.loc[(GD.index == s) & (GD['Img'] == i) ,'LocNotch2'] = locn2

                    # plotting
                    if SavedPlots:
                        RGBimg = io.imread(P + '\\' + s + '.tif', key = i)

                        fig0, [ax0, ax1] = plt.subplots(ncols = 2, dpi = 250,facecolor='black')
                        fig0.suptitle(s)
                        ax0.set_title('Attach point (magenta), notches (red),\n and center (blue) for alignement',fontsize=8)
                        ax0.imshow(RGBimg)
                        ax0.plot(Xsmooth,Ysmooth,'c',lw = 0.5)
                        ax0.plot(xN1,yN1,'.r',ms=2)
                        ax0.plot(xN2,yN2,'.r',ms=2)
                        ax0.plot(xA,yA,'.m',ms=2)
                        ax0.plot(Xcenter[0],Ycenter[0],'*b', ms = 3)
                        ax0.set_xticks([], [])
                        ax0.set_yticks([], [])
                        sc = ax1.scatter(Xsmooth,-Ysmooth+2*Ycenter[0], c= SignedCurv, cmap = 'BrBG', s = 2)
                        ax1.plot(xN1,-yN1+2*Ycenter[0],'ob',ms=3)
                        ax1.plot(xN2,-yN2+2*Ycenter[0],'ob',ms=3)
                        ax1.plot(xA,-yA+2*Ycenter[0],'sb',ms=3)
                        ax1.plot(Xsmooth[loc],-Ysmooth[loc]+2*Ycenter[0],'r+',mfc='none', ms = 3, lw= 0.5)
                        ax1.set_aspect('equal')
                        ax1.set_xticks([], [])
                        ax1.set_yticks([], [])
                        #ax1.set_xlim(left=-250+Xcenter[0],right=250+Xcenter[0])
                        #ax1.set_ylim(top=250+Ycenter[0],bottom=-250+Ycenter[0])
                        fig0.colorbar(sc, ax = ax1, label = 'Curvature',shrink = 0.6)
                        fig0.tight_layout()

                        fig0.savefig(P + '\\Figures\\Landmarks\\'+ s + '\\' + str(i) +'.png') 
                        plt.close()

                # Save tmp file with already analyzed stacks        
                GD.to_csv(P + '\\GlobalData' + stringName + '_Landmarks_tmp.csv',index_label = 'Ind')
                CD.to_csv(P + '\\ContourData' + stringName + '_Landmarks_tmp.csv',index_label = 'Ind')

            print('\n')

    return(CD,GD)

#%% getContourCurvature + fitCircle

def getContourCurvature(X,Y,step):

    newX = np.concatenate((X[-step:],X,X[:step]))    
    newY = np.concatenate((Y[-step:],Y,Y[:step]))
    
    Curv = []
    xc = []
    yc = []
    
    for i in range(len(X)):
        a = fitCircle(newX[i:i+2*step],newY[i:i+2*step])
        xc.append(a[0])
        yc.append(a[1])
        Curv.append(1/a[2])
        
    return(xc,yc,Curv)

def fitCircle(X,Y):
    
    x_m = np.mean(X)
    y_m = np.mean(Y)    
    
    def calc_R(xc, yc):

        return np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    
    def f_2(c):
        
        Ri = calc_R(*c)
        return Ri - Ri.mean()
    
    center_estimate = x_m, y_m
    center_fit, _ = optimize.leastsq(f_2, center_estimate)

    xc_fit, yc_fit = center_fit
    Ri_fit       = calc_R(xc_fit, yc_fit)
    
     #Fitting the radius of the circle
    R_fit        = Ri_fit.mean()
    
    if False:
        fig,ax = plt.subplots(dpi=250, facecolor = 'black')
        ax.plot(X,Y,'ro')

        xcircle = xc_fit + R_fit*np.cos(np.linspace(-np.pi,np.pi,100))
        ycircle = yc_fit + R_fit*np.sin(np.linspace(-np.pi,np.pi,100))
        ax.plot(xcircle,ycircle,'k--')

    return([xc_fit, yc_fit,R_fit])

#%% getContourPointsCoordinates

def getContourPointsCoordinates(P,s,npts,nimg,X,Y,Xc,Yc,Xold,Yold,Xcold,Ycold,Title):            
        
    RGBimg = io.imread(P + '\\' + s + '.tif', key = nimg) # get the  image from tiff stack

    # ask user to click
    get_ipython().run_line_magic('matplotlib', 'qt')
    f,ax = plt.subplots(dpi=250)
    ax.set_title(Title, color = 'black', fontsize=25)
    ax.imshow(RGBimg)
    ax.plot(X,Y,'c',lw = 0.7)
    ax.plot(Xold,Yold,'b--',lw=0.3)
    ax.plot(Xcold,Ycold,'mo',ms = 2)
    ax.plot(Xc,Yc,'r+',ms = 1.5)
    pts = np.asarray(plt.ginput(npts, timeout=-1))

    plt.close()
    get_ipython().run_line_magic('matplotlib', 'inline')

    Points = pts[:]

    for i in range(npts):
        x = pts[i][0]
        y = pts[i][1]

        xc = X[np.argmin(np.sqrt(np.square(X-x)+np.square(Y-y)))]            
        yc = Y[np.argmin(np.sqrt(np.square(X-x)+np.square(Y-y)))]

        Points[i][0] = xc
        Points[i][1] = yc

    return(Points)