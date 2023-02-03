# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 09:36:00 2022

@author: laplaud
"""


from StatsFunctions import plotSig, Corr,TwowayANOVA, StatsKruskal
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.stats import ks_2samp, linregress
from cycler import cycler
import VallapFunc as vf
from tqdm import tqdm

import gc

from matplotlib.patches import Rectangle

from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd

import os
#%% Data visualisation function for growth and osmotic chocs quantifications

# 1. Comparison of growth quantifications between different experiments

# 1.0 simple plot of curves

def PlotGrowth(GDs,FPH,Labels,colors):# Computing mean area over all gemmae for each image
    fig, ax = plt.subplots(dpi = 300)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Area (normalized)')
    for GD,col in zip(GDs,colors):    
        nppg = len(np.unique(GD.index))
        nimgmax = GD['Img'].max()
        MeanA = np.empty(nimgmax)
        MeanTime = np.empty(nimgmax)
        StdA = np.empty(nimgmax)

        for im in range(nimgmax):
            MeanA[im] = GD.loc[GD['Img'] == im,'Area'].to_numpy().mean()
            MeanTime[im] = im*60/FPH
            StdA[im] = GD.loc[GD['Img'] == im,'Area'].to_numpy().std()
            
        ax.errorbar(MeanTime,MeanA/MeanA[0],yerr=StdA/MeanA[0]/np.sqrt(nppg), color = col, capsize=3)
        
    ax.legend(Labels)

# 1.1 plot curves and quantif

# GDs : list of dataframe global data for experiments to analyse, Labels : list of 
# labels for each experiment, colors list of colors to plot, P : path for figure saving,
# Title : Name of the conditions plot (appears in figure titles)

# Kwargs : 'showcurve/box/hist' (True/False) to choose the different graphs displayed,
# 'indiplots' : (True/False) plot individual area curves, 'nbins' : (integer) 
# number of bins in histograms, 'stats' ('ranksum' or None) choose type of stats done 
# 'sigpairs' (list of 2-lists of integers) if specified, will only do significance 
# tests between selected data sets

def compareGrowth(GDs, Labels, colors,P, Title, **kwargs):

    showcurve = False
    showbox = False
    showhist = False
    nbins = 20
    AllSigs = True
    IndividualPlots = False
    stats = 'ranksum'
    groupcat = None
    diffcat = None
    NimgMax = '24h'
    
    for key, value in kwargs.items(): 
        if key == 'showcurve':
            showcurve = value 
        elif key == 'showbox':
            showbox = value
        elif key == 'showhist':
            showhist = value
        elif key == 'nbins':
            nbins = value
        elif key == 'sigpairs':
            sigpairs = value
            AllSigs = False
        elif key == 'indiplots' :
            IndividualPlots = value
        elif key == 'stats' :
            stats = value
        elif key == 'groupcat' :
            groupcat = np.array(value)
        elif key == 'diffcat' :
            diffcat = np.array(value)
        elif key == 'NimgMax' :
            if (value == '24h') | (value == 'max'):
                NimgMax = value 
            else:
                raise ValueError('Wrong value for NimgMax ! Allowed : ''24h'' or ''max'' ')
                
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    # Colors
    colorcycle = [plt.get_cmap('gist_rainbow')(1. * i/len(GDs)) for i in range(len(GDs))]
    mpl.rcParams['axes.prop_cycle'] = cycler(color=colorcycle)
         
    # check existence of figure folder, if absent, create it
    if not os.path.exists(P):
            os.mkdir(P) # create folder
            
    # check existence of figure folder, if absent, create it
    if not os.path.exists(P + '\\AreaGrowth'):
            os.mkdir(P + '\\AreaGrowth') # create folder
            
    ## Data grouping if ANOVA : 
    if stats == 'ANOVA':
        cats = np.unique(groupcat)
        newGDs = [None]*len(cats)
        for s,ss in zip(cats,range(len(cats))):
            pos = np.argwhere(groupcat==s)
            PD = pd.DataFrame(data=None)
            for i in pos:
                PD = PD.append(GDs[i[0]])
            newGDs[ss] = PD[:]
    else:
        newGDs = GDs[:]
    
    ######### Curves of evolution ##########
    
    fig2,ax2 = plt.subplots(dpi = 250,facecolor='black')
    fig2.suptitle(Title + ' - Area vs. time')
    plt.xlabel('Time (min)')
    plt.ylabel('Area (mm²)')
    
    fig3,ax3 = plt.subplots(dpi = 250,facecolor='black')
    fig3.suptitle(Title + ' - Norm Area vs. time')
    plt.xlabel('Time (min)')
    plt.ylabel('Area (normalized)')
    
    fig4,ax4 = plt.subplots(dpi = 250,facecolor='black')
    fig4.suptitle(Title + ' - dAdt vs. time')
    plt.xlabel('Time (min)')
    plt.ylabel('Area derivative (µm²/min)')
    
    fig5,ax5 = plt.subplots(dpi = 250,facecolor='black')
    fig5.suptitle(Title + ' - GrowthRate vs. time')
    plt.xlabel('Time (min)')
    plt.ylabel('Growth rate (day-1)')
    
    fig6,ax6 = plt.subplots(dpi = 250,facecolor='black')
    fig6.suptitle(Title + ' - GrowthRate aligned vs. time')
    plt.xlabel('Time since Tstart (min)')
    plt.ylabel('Growth rate aligned at tdeb (day-1)')
    
    
    fig41,ax41 = plt.subplots(dpi = 250,facecolor='black')
    fig41.suptitle(Title + ' - dAdt (Time avg) vs. A (Time avg)')
    plt.xlabel('Area (mm²)')
    plt.ylabel('Area derivative (µm²/min)')
    
    fig51,ax51 = plt.subplots(dpi = 250,facecolor='black')
    fig51.suptitle(Title + ' - GrowthRate (Time avg) vs. A (Time avg)')
    plt.xlabel('Area (mm²)')
    plt.ylabel('Growth rate (day-1)')
    
    fig52,ax52 = plt.subplots(dpi = 250,facecolor='black')
    fig52.suptitle(Title + ' - GrowthRate (Time avg) vs. Anorm (Time avg)')
    plt.xlabel('Area (norm)')
    plt.ylabel('Growth rate (day-1)')
    
    
    
    fig42,ax42 = plt.subplots(dpi = 250,facecolor='black')
    fig42.suptitle(Title + ' - dAdt (Area avg, a) vs. A (Area avg, a)')
    plt.xlabel('Area (mm²)')
    plt.ylabel('Area derivative (µm²/min)')
    
    fig53,ax53 = plt.subplots(dpi = 250,facecolor='black')
    fig53.suptitle(Title + ' - GrowthRate (Area avg, a) vs. A (Area avg, a)')
    plt.xlabel('Area (mm²)')
    plt.ylabel('Growth rate (day-1)')
    
    
    
    fig43,ax43 = plt.subplots(dpi = 250,facecolor='black')
    fig43.suptitle(Title + ' - dAdt (Area avg, b) vs. A (Area avg, b)')
    plt.xlabel('Area (mm²)')
    plt.ylabel('Area derivative (µm²/min)')
    
    fig54,ax54 = plt.subplots(dpi = 250,facecolor='black')
    fig54.suptitle(Title + ' - GrowthRate (Area avg, b) vs. A (Area avg, b)')
    plt.xlabel('Area (mm²)')
    plt.ylabel('Growth rate (day-1)')

    
    for GD,lab,i in zip(newGDs,Labels,range(len(GDs))):
        
        StackList = np.unique(GD.index)
           
        if NimgMax == 'max':
            nimgmax = GD['Img'].max() # number of images (duration) to plot for growth curve
        else:
            nimgmax = 49 # 24h
        
        if IndividualPlots:
            fig1,ax1 = plt.subplots(dpi = 250,facecolor='black')
            fig1.suptitle(lab + ' - Area vs. time')
            plt.xlabel('Time (min)')
            plt.ylabel('Area (mm²)')
            for s in StackList:
                ax1.plot(GD.loc[s,'Img']*30,GD.loc[s,'Area'],label=s,lw=1)
            plt.legend(prop={'size': 5})

        # number of ppgs and label
        nPPG = len(GD.loc[GD['Img'] == 0])
        lab = lab + ' - n = ' + str(nPPG)
        
        # Computing averages over all gemmae for each image
        MeanA = np.empty(nimgmax)
        MeandAdt = np.empty(nimgmax)
        MeanTime = np.empty(nimgmax)
        StdA = np.empty(nimgmax)
        StddAdt = np.empty(nimgmax)
        MeanGR = np.empty(nimgmax)
        StdGR = np.empty(nimgmax)
        MeanGRal = np.empty(nimgmax)
        StdGRal = np.empty(nimgmax)

        for im in range(nimgmax):

            MeanA[im] = GD.loc[GD['Img'] == im,'Area'].to_numpy().mean()
            MeandAdt[im] = np.nanmean(GD.loc[GD['Img'] == im,'dAdt'].to_numpy()*1000000) # µm²/min
            MeanTime[im] = im*30
            StdA[im] = GD.loc[GD['Img'] == im,'Area'].to_numpy().std()
            StddAdt[im] = np.nanstd(GD.loc[GD['Img'] == im,'dAdt'].to_numpy()*1000000)
            MeanGR[im] = GD.loc[GD['Img'] == im,'GR_Full'].to_numpy().mean()*60*24
            StdGR[im] = GD.loc[GD['Img'] == im,'GR_Full'].to_numpy().std()*60*24
            MeanGRal[im] = np.nanmean(GD.loc[GD['Img'] == im,'GR_Full_al'].to_numpy())
            StdGRal[im] = np.nanstd(GD.loc[GD['Img'] == im,'GR_Full_al'].to_numpy())  
                
        nppg = len(GD.loc[GD['Img'] == 0,'Area'].to_numpy())



        ### Computing averages based on size rather than time
        Areas = GD['Area'].to_numpy() # mm²
        GrowthRates = GD['GR_Full'].to_numpy()*60*24 # day-1
        dAdts = GD['dAdt'].to_numpy()*1000000 # µm²/min
        
        npts = len(Areas)
        
        # Averages with constant number of elements
        
        mapping = np.argsort(Areas)
        
        Areas_Sort = Areas[mapping]
        GrowthRates_Sort = GrowthRates[mapping]
        dAdts_Sort = dAdts[mapping]
        
        ncurvepts = 15 
        
        samplesize = int(np.ceil(npts/ncurvepts))
        
        aMeanA = np.empty(ncurvepts)
        aMeandAdt = np.empty(ncurvepts)
        aStdA = np.empty(ncurvepts)
        aStddAdt = np.empty(ncurvepts)
        aMeanGR = np.empty(ncurvepts)
        aStdGR = np.empty(ncurvepts)
        
        for ia in range(ncurvepts):
            
            aMeanA[ia] = np.nanmean(Areas_Sort[ia*samplesize:(ia+1)*samplesize-1])
            aMeanGR[ia] = np.nanmean(GrowthRates_Sort[ia*samplesize:(ia+1)*samplesize-1])
            aMeandAdt[ia] = np.nanmean(dAdts_Sort[ia*samplesize:(ia+1)*samplesize-1])
            aStdA[ia] = np.nanstd(Areas_Sort[ia*samplesize:(ia+1)*samplesize-1].std())
            aStdGR[ia] = np.nanstd(GrowthRates_Sort[ia*samplesize:(ia+1)*samplesize-1])
            aStddAdt[ia] = np.nanstd(dAdts_Sort[ia*samplesize:(ia+1)*samplesize-1])
           
        
        # Averages with constant bin size
        
        binsize = 0.05 # mm²
        
        # nbin = int(np.ceil(np.max(Areas)/binsize))
        nbin = int(np.ceil(0.9/binsize))
        
        bMeanA = np.empty(nbin)
        bMeandAdt = np.empty(nbin)
        bStdA = np.empty(nbin)
        bStddAdt = np.empty(nbin)
        bMeanGR = np.empty(nbin)
        bStdGR = np.empty(nbin)
        sqrtn = np.empty(nbin)
        
        
        for ib in range(nbin):
            
            mask = (Areas>=ib*binsize)&(Areas<(ib+1)*binsize)
            
            sqrtn[ib] = np.sqrt(np.sum(mask))
            
            bMeanA[ib] = np.nanmean(Areas[mask])
            bStdA[ib] = np.nanstd(Areas[mask])
            bMeandAdt[ib] = np.nanmean(dAdts[mask])
            bStddAdt[ib] = np.nanstd(dAdts[mask])
            bMeanGR[ib] = np.nanmean(GrowthRates[mask])
            bStdGR[ib] = np.nanstd(GrowthRates[mask])
        
        # Plots
        
        ax2.errorbar(MeanTime,MeanA,yerr=StdA/np.sqrt(nppg), capsize=3,label=lab,color = colors[i])
        ax3.errorbar(MeanTime,MeanA/MeanA[0],yerr=StdA/MeanA[0]/np.sqrt(nppg), capsize=3,label=lab,color = colors[i])
        ax4.errorbar(MeanTime,MeandAdt,yerr=StddAdt/np.sqrt(nppg), capsize=3,label=lab,color = colors[i])
        ax5.errorbar(MeanTime,MeanGR,yerr=StdGR/np.sqrt(nppg), capsize=3,label=lab,color = colors[i])
        ax6.errorbar(MeanTime,MeanGRal,yerr=StdGRal/np.sqrt(nppg), capsize=3,label=lab,color = colors[i])
        
        ax41.errorbar(MeanA,MeandAdt, yerr=StddAdt/np.sqrt(nppg), capsize=3,label=lab,color = colors[i])
        ax51.errorbar(MeanA,MeanGR,yerr=StdGR/np.sqrt(nppg), capsize=3,label=lab,color = colors[i])
        ax52.errorbar(MeanA/MeanA[0],MeanGR,yerr=StdGR/np.sqrt(nppg), capsize=3,label=lab,color = colors[i])
        
        ax42.errorbar(aMeanA,aMeandAdt, yerr = aStddAdt/np.sqrt(50), capsize=3,label=lab,color = colors[i])
        ax53.errorbar(aMeanA,aMeanGR,yerr = aStdGR/np.sqrt(50), capsize=3,label=lab,color = colors[i])
        
        ax43.errorbar(bMeanA,bMeandAdt, yerr = bStddAdt/sqrtn, capsize=3,label=lab,color = colors[i])
        ax54.errorbar(bMeanA,bMeanGR,yerr = bStdGR/sqrtn, capsize=3,label=lab,color = colors[i])
        
        
        
    plt.figure(fig2.number)
    plt.legend(prop={'size': 8})
    plt.figure(fig3.number)
    plt.legend(prop={'size': 8})
    plt.figure(fig4.number)
    plt.legend(prop={'size': 8})
    plt.figure(fig5.number)
    plt.legend(prop={'size': 8})
    plt.figure(fig6.number)
    plt.legend(prop={'size': 8})
    plt.figure(fig41.number)
    plt.legend(prop={'size': 8})
    plt.figure(fig51.number)
    plt.legend(prop={'size': 8})
    plt.figure(fig52.number)
    plt.legend(prop={'size': 8})
    plt.figure(fig42.number)
    plt.legend(prop={'size': 8})
    plt.figure(fig53.number)
    plt.legend(prop={'size': 8})
    plt.figure(fig43.number)
    plt.legend(prop={'size': 8})
    plt.figure(fig54.number)
    plt.legend(prop={'size': 8})
    
    if not showcurve:
        plt.close(fig2)
        plt.close(fig3)
        plt.close(fig4)
        plt.close(fig5)
        plt.close(fig6)
        plt.close(fig41)
        plt.close(fig51)
        plt.close(fig52)
        plt.close(fig42)
        plt.close(fig53)
        plt.close(fig43)
        plt.close(fig54)


    ######### Parameters of fit ###########
    
      
    fig4,ax4 = plt.subplots(dpi = 250,facecolor='black')
    fig4.suptitle(Title + ' - Growth start time')
      
    fig5,ax5 = plt.subplots(dpi = 250,facecolor='black')
    fig5.suptitle(Title + ' - Growth caracteristic time')
      
    fig51,ax51 = plt.subplots(dpi = 250,facecolor='black')
    fig51.suptitle(Title + ' - Final Growth Rate')
    
    fig6,ax6 = plt.subplots(dpi = 250,facecolor='black') 
    fig6.suptitle(Title + ' - Starting area') 

    fig16,ax16 = plt.subplots(dpi = 250,facecolor='black')
    fig16.suptitle(Title + ' - Initial growth increase')
    
    if len(newGDs) == 2:
        # Histogram for distribution comparison
        fig7,ax7 = plt.subplots(dpi = 250,figsize = (5,3.5),facecolor='black')
        fig7.suptitle(Title + ' - Growth caracteristic times')
        plt.xlabel('Tau growth (hours)')
        plt.ylabel('PDF')

        fig8,ax8 = plt.subplots(dpi = 250,figsize = (5,3.5),facecolor='black')
        fig8.suptitle(Title + ' - Growth start time')
        plt.xlabel('T start (min)')
        plt.ylabel('PDF')
        
        fig9,ax9 = plt.subplots(dpi = 250,figsize = (5,3.5),facecolor='black')
        fig9.suptitle(Title + ' - Growth caracteristic times')
        plt.xlabel('Tau growth (hours) - median aligned')
        plt.ylabel('PDF')

        fig10,ax10 = plt.subplots(dpi = 250,figsize = (5,3.5),facecolor='black')
        fig10.suptitle(Title + ' - Growth start time')
        plt.xlabel('T start (min) - median aligned')
        plt.ylabel('PDF')
    
    tdebs= [None]*len(newGDs)
    taus= [None]*len(newGDs)
    captdeb= [None]*len(newGDs)
    captau= [None]*len(newGDs)
    medtdeb= [None]*len(newGDs)
    medtau= [None]*len(newGDs)    
    
    GR_ends = [None]*len(newGDs)
    capGR_ends = [None]*len(newGDs)
    medGR_ends = [None]*len(newGDs)   
    
    Area0 = [None]*len(newGDs) 
    capArea0 = [None]*len(newGDs) 
    medArea0 = [None]*len(newGDs) 
    
    AreaStart = [None]*len(newGDs)
    capAreaStart = [None]*len(newGDs)
    medAreaStart = [None]*len(newGDs)
    
    grouping = []
    labs = []
    
    for GD,lab,i in zip(newGDs,Labels,range(len(newGDs))):
        
        # number of ppgs and label
        nPPG = len(GD.loc[GD['Img'] == 0])
        lab = lab + 'n = ' + str(nPPG)
        labs = np.append(labs,lab)
        
        # Retrieve data
        tdebs[i] = GD.loc[GD['Img'] == 0, 'tdeb']/60
        taus[i] = GD.loc[GD['Img'] == 0, 'Tau']/60          
        GR_ends[i] = GD.loc[GD['Img'] == 0, 'GR_end']  # in days-1     
        Area0[i] = GD.loc[GD['Img'] == 0, 'A0fit'] 
        AreaStart[i] = GD.loc[GD['Img'] == 0, 'GrowthAtStart_flat']*100
        
        
        # swarmplots
        grouping = np.append(grouping,np.ones(len(tdebs[i]))*i)


        plotprops = {'color':'white'}
        boxprops = {'color':'white','facecolor':colors[i]}
        
         
        bp4 = ax4.boxplot(tdebs[i], positions = [i], labels = [lab],patch_artist = True, boxprops=boxprops, capprops =plotprops,
                    showfliers = False,whiskerprops=plotprops,medianprops =plotprops)
        
        bp5 = ax5.boxplot(taus[i], positions = [i], labels = [lab],patch_artist = True, boxprops=boxprops, capprops =plotprops,
                    showfliers = False,whiskerprops=plotprops,medianprops =plotprops)
        
        bp51 = ax51.boxplot(GR_ends[i], positions = [i], labels = [lab],patch_artist = True, boxprops=boxprops, capprops =plotprops,
                    showfliers = False,whiskerprops=plotprops,medianprops =plotprops)
        
        bp6 = ax6.boxplot(Area0[i], positions = [i], labels = [lab],patch_artist = True, boxprops=boxprops, capprops =plotprops, 
            showfliers = False,whiskerprops=plotprops,medianprops =plotprops) 
        
        bp26 = ax16.boxplot(AreaStart[i], positions = [i], labels = [lab],patch_artist = True, boxprops=boxprops, capprops =plotprops,
                    showfliers = False,whiskerprops=plotprops,medianprops =plotprops)
    
        captdeb[i] = bp4['caps'][1].get_ydata(orig=True)[0]
        captau[i] = bp5['caps'][1].get_ydata(orig=True)[0]
        capGR_ends[i] = bp51['caps'][1].get_ydata(orig=True)[0]
        capArea0[i] = bp6['caps'][1].get_ydata(orig=True)[0] 
        capAreaStart[i] = bp26['caps'][1].get_ydata(orig=True)[0]
        medtdeb[i] = bp4['medians'][0].get_ydata(orig=True)[0]
        medtau[i] = bp5['medians'][0].get_ydata(orig=True)[0]
        medGR_ends[i] = bp51['medians'][0].get_ydata(orig=True)[0]
        medArea0[i] = bp6['medians'][0].get_ydata(orig=True)[0] 
        medAreaStart[i] = bp26['medians'][0].get_ydata(orig=True)[0]
        
        if len(newGDs) == 2:
        
            ax7.hist(taus[i], nbins, density=True, facecolor=colors[i], alpha=0.5)
            ax8.hist(tdebs[i], nbins, density=True, facecolor=colors[i], alpha=0.5)
            ax9.hist(taus[i]-np.median(taus[i]), nbins, density=True, facecolor=colors[i], alpha=0.5)
            ax10.hist(tdebs[i]-np.median(tdebs[i]), nbins, density=True, facecolor=colors[i], alpha=0.5)

            
    sns.swarmplot(x=grouping,y=pd.concat(tdebs),color = 'white', size=2, ax = ax4)
    sns.swarmplot(x=grouping,y=pd.concat(taus),color = 'white', size=2, ax = ax5)
    sns.swarmplot(x=grouping,y=pd.concat(GR_ends),color = 'white', size=2, ax = ax51)
    sns.swarmplot(x=grouping,y=pd.concat(Area0),color = 'white', size=2, ax = ax6) 
    sns.swarmplot(x=grouping,y=pd.concat(AreaStart),color = 'white', size=2, ax = ax16)
    
    ax4.set_xticklabels(labs)
    ax5.set_xticklabels(labs)
    ax51.set_xticklabels(labs)
    ax6.set_xticklabels(labs) 
    ax16.set_xticklabels(labs)

    if len(newGDs) == 2:
        # Distribution comparison with two-sample kolmogorov smirnov test
        statsTau, pTau =  ks_2samp(taus[0],taus[1])
        ax7.set_title('KS test - p = ' + str(round(pTau*1000)/1000))
        fig7.tight_layout()
        
        statsTdeb, pTdeb =  ks_2samp(tdebs[0],tdebs[1])
        ax8.set_title('KS test - p = ' + str(round(pTdeb*1000)/1000))
        fig8.tight_layout()
        
        statsTau, pTau =  ks_2samp(taus[0]-np.median(taus[0]),taus[1]-np.median(taus[1]))
        ax9.set_title('KS test - p = ' + str(round(pTau*1000)/1000))
        fig9.tight_layout()
        
        statsTdeb, pTdeb =  ks_2samp(tdebs[0]-np.median(tdebs[0]),tdebs[1]-np.median(tdebs[1]))
        ax10.set_title('KS test - p = ' + str(round(pTdeb*1000)/1000))
        fig10.tight_layout()
        
        if not showhist:
            plt.close(fig7)
            plt.close(fig8)
            plt.close(fig9)
            plt.close(fig10)
    
    steptdeb = np.max(captdeb)*0.125
    steptau = np.max(captau)*0.125
    stepGR_ends = np.max(capGR_ends)*0.125
    stepArea0 = np.max(capArea0)*0.125 
    stepAreaStart = np.max(capAreaStart)*0.125
    
    fullsteptdeb = 0
    fullsteptau = 0
    fullstepGR_ends = 0
    fullstepArea0 = 0 
    fullstepAreaStart = 0
    
    hmaxtdeb = np.max(captdeb)
    hmaxtau = np.max(captau)
    hmaxGR_ends = np.max(capGR_ends)
    hmaxArea0 = np.max(capArea0) 
    hmaxAreaStart = np.max(capAreaStart)
    
    ax4.set_ylabel('Tstart (hours)')
    ax5.set_ylabel('Tau growth (hours)')
    ax51.set_ylabel('Final growth rate  (days-1)')
    ax6.set_ylabel('Starting area from fit (mm²)') 
    ax16.set_ylabel('Growth at Tstart (%)')
    
    if stats=='ranksum':
        if AllSigs:
            for i in range(len(newGDs)-1):
                for j in range(i+1,len(newGDs)):

                    fullsteptdeb = plotSig(ax4,hmaxtdeb,steptdeb,fullsteptdeb,tdebs[i],tdebs[j],i,j)

                    fullsteptau = plotSig(ax5,hmaxtau,steptau,fullsteptau,taus[i],taus[j],i,j)

                    fullstepGR_ends = plotSig(ax51,hmaxGR_ends,stepGR_ends,fullstepGR_ends,GR_ends[i],GR_ends[j],i,j)
                    
                    fullstepArea0 = plotSig(ax6,hmaxArea0,stepArea0,fullstepArea0,Area0[i],Area0[j],i,j) 

                    fullstepAreaStart = plotSig(ax16,hmaxAreaStart,stepAreaStart,fullstepAreaStart,AreaStart[i],AreaStart[j],i,j)
                    
        else:
            for i,j in sigpairs:

                fullsteptdeb = plotSig(ax4,hmaxtdeb,steptdeb,fullsteptdeb,tdebs[i],tdebs[j],i,j)

                fullsteptau = plotSig(ax5,hmaxtau,steptau,fullsteptau,taus[i],taus[j],i,j)

                fullstepGR_ends = plotSig(ax51,hmaxGR_ends,stepGR_ends,fullstepGR_ends,GR_ends[i],GR_ends[j],i,j)
 
                fullstepArea0 = plotSig(ax6,hmaxArea0,stepArea0,fullstepArea0,Area0[i],Area0[j],i,j) 

                fullstepAreaStart = plotSig(ax16,hmaxAreaStart,stepAreaStart,fullstepAreaStart,AreaStart[i],AreaStart[j],i,j)

    elif stats == 'kruskal':
        
        StatsKruskal(ax4,tdebs)
        StatsKruskal(ax5,taus)
        StatsKruskal(ax51,GR_ends)
        StatsKruskal(ax6,Area0)
        StatsKruskal(ax16,AreaStart)
               
    
 
    if stats=='ranksum':
        # fig4.savefig(P + '\\AreaGrowth\\' + Title + '_Tstart.png')
        # fig5.savefig(P + '\\AreaGrowth\\'+ Title +  '_TauGrowth.png')
        # fig6.savefig(P + '\\AreaGrowth\\'+ Title +  '_StartingArea.png') 
        # fig16.savefig(P + '\\AreaGrowth\\'+ Title +  '_InitialGrowth.png')
        if not showbox:
            plt.close(fig5)
            plt.close(fig51)
            plt.close(fig4)
            plt.close(fig6)
            plt.close(fig16)
        return  
    elif stats == 'ANOVA':
        for v,med,fig,ax,dat in zip(['tdeb','Tau','A0fit','GrowthAtStart_flat'],[medtdeb,medtau,medArea0,medAreaStart],[fig4,fig5,fig6,fig16],[ax4,ax5,ax6,ax16],
                                    [pd.concat(tdebs),pd.concat(taus),pd.concat(Area0),pd.concat(AreaStart)]):

            res = TwowayANOVA(v,diffcat,groupcat,GDs);

            pDC = res.loc['C(Date)','PR(>F)']
            pGC = res.loc['C(Condition)','PR(>F)']

            ax.set_title('Date significativity (ANOVA) : p = ' + str(round(pDC*1000)/1000))
            ax.plot([0.3, 0.7],[np.mean(med), np.mean(med)],'-w')
            ax.text(0.5,np.mean(med)*1.1,'p = ' + str(round(pGC*1000)/1000), ha='center',fontsize='small')
            
            ax.set_ylim([np.min(dat)-np.abs(0.3*np.min(dat)), 1.5*np.percentile(dat,90)])
            
            
    
    else:
        if not showbox:
            plt.close(fig5)
            plt.close(fig51)
            plt.close(fig4)
            plt.close(fig6) 
            plt.close(fig16)
            return
        else:
            return


# 2. Comparison of hydromechanical properties

# GDs : list of dataframe global data for experiments to analyse, Labels : list of 
# labels for each experiment, colors list of colors to plot, P : path for figure saving,
# Title : Name of the conditions plot (appears in figure titles)

# Kwargs : 'showbox/hist' (True/False) to choose the different graphs displayed,
# 'stats' ('ranksum' or None) choose type of stats done 
# 'sigpairs' (list of 2-lists of integers) if specified, will only do significance 
# tests between selected data sets

def compareHydroMech(GDs, Labels, colors,P, Title, **kwargs):

    showbox = False
    showhist = False
    showE = True
    showTau = False
    AllSigs = True
    stats = 'ranksum'
    indiplots = True
    
    for key, value in kwargs.items(): 
        if key == 'showbox':
            showbox = value
        elif key == 'showhist':
            showhist = value
        elif key == 'showE':
            showE = value
        elif key == 'showTau':
            showTau = value
        elif key == 'sigpairs':
            sigpairs = value
            AllSigs = False
        elif key == 'stats' :
            stats = value
        elif key == 'indiplots' :
            indiplots = value
        else:
            print('Unknown key : ' + key + '. Ewarg ignored.')
         
    # check existence of figure folder, if absent, create it
    if not os.path.exists(P + '\\Hydromechanics'):
            os.mkdir(P + '\\Hydromechanics') # create folder
    
        
    ### Regroup data
    Ecomps= [None]*len(GDs)
    TauComps= [None]*len(GDs)
    Erels= [None]*len(GDs)
    TauRels= [None]*len(GDs)
    LovHs = [None]*len(GDs) 
    Phis = [None]*len(GDs)
    
    AllRatios = np.empty(0)
    
    n = len(GDs)
    
    # Figure for E ratios 
    f3,ax3 = plt.subplot_mosaic(vf.mosaicList(n)[0], dpi=200, figsize=(7,5))
    f3.patch.set_facecolor('black')
    ax3['a'].set_title('Growth rates change\n caused by Osmotic choc')
    
    
    
    for GD,lab,i,nax in zip(GDs,Labels,range(len(GDs)),vf.mosaicList(n)[1]):
        
        # Retrieve data
        Ecomps[i] = GD.loc[GD['Img'] == 0, 'Ecomp']
        TauComps[i] = GD.loc[GD['Img'] == 0, 'TauFlux'] 
        Erels[i] = GD.loc[GD['Img'] == 0, 'Erel']
        TauRels[i] = GD.loc[GD['Img'] == 0, 'TauFluxRel']  
        LovHs[i] = GD.loc[GD['Img'] == 0, 'L/H0'] 
        Phis[i] = GD.loc[GD['Img'] == 0, 'Phi']*60
        
        
        Eratios = np.divide(Erels[i],Ecomps[i])
        AllRatios = np.append(AllRatios,Eratios)
        
        if indiplots:
            fig0,ax0,cap,med = vf.boxswarmplot(Title + '\n\nElastic bulk modulus comparison for ' + lab,'E (MPa)',
                                               [Ecomps[i],Erels[i]],[colors[i],colors[i]],['Ed','Ei'])
    
            plotSig(ax0,np.max(cap),np.max(cap)*0.125,0,Ecomps[i],Erels[i],0,1)
            
            fig0.tight_layout() 
            # fig0.savefig(P + '\\Hydromechanics\\' + lab + '_EComp-Rel.png')
            if not showE:
                plt.close(fig0)
            
            fig01,ax01,cap,med = vf.boxswarmplot(Title + '\n\nCracteristic time comparison for ' + lab,'Tau (min)',
                                               [TauComps[i],TauRels[i]],[colors[i],colors[i]],['Tau_comp','Tau_rel'])
    
            plotSig(ax01,np.max(cap),np.max(cap)*0.125,0,TauComps[i],TauRels[i],0,1)
            
            fig01.tight_layout()
            # fig01.savefig(P + '\\Hydromechanics\\' + lab + '_L/H_Comp-Rel.png')
            if not showTau:
                plt.close(fig01)
            
            
            if showhist:
                fig00, ax00 = plt.subplots(dpi=300)
                ax00.hist(Eratios, facecolor=colors[i]) # ,density = True
                fig00.suptitle('Median : ' + str(np.round(Eratios.median()*100)/100) + 
                               ' - Mean : ' + str(np.round(Eratios.mean()*100)/100))
                ax00.set_xlabel('Ei/Ed')
                ax00.set_ylabel('Count')
                # fig00.savefig(P + '\\Hydromechanics\\' + lab + '_EComp-Rel_Dist.png')
                if not showE:
                    plt.close(fig00)
                
                
    
                linreg = linregress(Ecomps[i],Erels[i])
    
                g = sns.jointplot(x=Ecomps[i],y=Erels[i],kind='reg',color = colors[i],height = 12)
                g.ax_joint.set_xlabel('E compression (MPa)',fontsize = 25)
                g.ax_joint.set_ylabel('E relaxation (MPa)',fontsize = 25)
                g.ax_joint.tick_params(axis='both', labelsize=20)
                g.ax_joint.legend([f"S = {linreg.slope:.2f}",
                                   f"CC = {linreg.rvalue:.3f}\nP = {linreg.pvalue:.3f}"],
                                  fontsize='xx-large')
                if not showE:
                    plt.close(g)
                
                fig001, ax001 = plt.subplots(dpi=300)
                ax001.hist(np.divide(TauRels[i],TauComps[i]), facecolor=colors[i]) # ,density = True
                fig001.suptitle('Median : ' + str(np.round(np.divide(TauRels[i],TauComps[i]).median()*100)/100) + 
                               ' - Mean : ' + str(np.round(np.divide(TauRels[i],TauComps[i]).mean()*100)/100))
                ax001.set_xlabel('Tau_Rel/Tau_Comp')
                ax001.set_ylabel('Count')
                # fig001.savefig(P + '\\Hydromechanics\\' + lab + '_L/HComp-Rel_Dist.png')
                if not showTau:
                    plt.close(fig001)
                
        if showhist:   
            ax3[nax].hist(Eratios, facecolor=colors[i], density = True)
            sns.kdeplot(Eratios,ax=ax3[nax], color = 'lightgray',lw=1)
             
            sns.kdeplot(Eratios,ax=ax3['a'],color = colors[i], label= lab)
            ax3[nax].set_ylabel('')
            ax3[nax].set_xlabel(lab)
    
    ### E ratios histograms
    ax3['a'].hist(AllRatios,color='gray', density = True, label = 'Pooled data')
    ax3['a'].set_ylabel('Density')
    ax3['a'].set_xlabel('Ei/Ed')
    ax3['a'].set_title('Mean : ' + '{0:.2f}'.format(AllRatios.mean()))
    ax3['a'].legend()
    f3.tight_layout()
    
    
    
    ### boxplots
    fig1,ax1,capEcomp,medEcomp = vf.boxswarmplot(Title + '\n\nElastic bulk modulus (compression)','Ed (MPa)',Ecomps,colors,Labels[:])
    fig10,ax10,capErel,medErel = vf.boxswarmplot(Title + '\n\nElastic bulk modulus (relaxation)','Ei (MPa)',Erels,colors,Labels[:])
    fig2,ax2,capTauComp,medTauComp = vf.boxswarmplot(Title + '\n\nCaracteristic time (compression)','Tau (min)',TauComps,colors,Labels[:])
    fig20,ax20,capTauRel,medTauRel = vf.boxswarmplot(Title + '\n\nCaracteristic time (relaxation)','Tau (min)',TauRels,colors,Labels[:]) 
    fig3,ax3,capLovH,medLovH = vf.boxswarmplot(Title + '\n\nConductivity','L/H0 (Pa-1.s-1)',LovHs,colors,Labels[:]) 
    fig4,ax4,capPhi,medPhi = vf.boxswarmplot(Title + '\n\nExtensibility','Phi (MPa-1.h-1)',Phis,colors,Labels[:])       


    ### stats
    fullstepE = 0
    fullstepEcomp = 0
    fullstepTauComp = 0
    fullstepErel = 0
    fullstepTauRel = 0
    
    if stats=='ranksum':
        if AllSigs:
            for i in range(len(GDs)-1):
                for j in range(i+1,len(GDs)):

                    fullstepEcomp = plotSig(ax1,np.max(capEcomp),np.max(capEcomp)*0.125,fullstepEcomp,Ecomps[i],Ecomps[j],i,j)
                    fullstepErel = plotSig(ax10,np.max(capErel),np.max(capErel)*0.125,fullstepErel,Erels[i],Erels[j],i,j)
                    fullstepTauComp = plotSig(ax2,np.max(capTauComp),np.max(capTauComp)*0.125,fullstepTauComp,TauComps[i],TauComps[j],i,j)
                    fullstepTauRel = plotSig(ax20,np.max(capTauRel),np.max(capTauRel)*0.125,fullstepTauRel,TauRels[i],TauRels[j],i,j)

        else:
            for i,j in sigpairs:

                    fullstepEcomp = plotSig(ax1,np.max(capEcomp),np.max(capEcomp)*0.125,fullstepEcomp,Ecomps[i],Ecomps[j],i,j)
                    fullstepErel = plotSig(ax10,np.max(capErel),np.max(capErel)*0.125,fullstepErel,Erels[i],Erels[j],i,j)
                    fullstepTauComp = plotSig(ax2,np.max(capTauComp),np.max(capTauComp)*0.125,fullstepTauComp,TauComps[i],TauComps[j],i,j)
                    fullstepTauRel = plotSig(ax20,np.max(capTauRel),np.max(capTauRel)*0.125,fullstepTauRel,TauRels[i],TauRels[j],i,j)

    fig1.tight_layout()
    fig2.tight_layout()
    fig10.tight_layout()
    fig20.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    
    

    if stats=='ranksum':
        # fig1.savefig(P + '\\Hydromechanics\\' + Title + '_Ecomp.png')
        # fig2.savefig(P + '\\Hydromechanics\\'+ Title +  '_TauFluxComp.png')
        # fig10.savefig(P + '\\Hydromechanics\\' + Title + '_Erel.png')
        # fig20.savefig(P + '\\Hydromechanics\\'+ Title +  '_TauFluxrel.png')
        # fig11.savefig(P + '\\Hydromechanics\\' + Title + '_E.png')
        if not showbox:
            plt.close(fig2)
            plt.close(fig1)
            plt.close(fig20)
            plt.close(fig10)
        else:
            if not showE:
                plt.close(fig1)
                plt.close(fig10)
            if not showTau:
                plt.close(fig2)
                plt.close(fig20)
            
        return    
    else:
        if not showbox:
            plt.close(fig2)
            plt.close(fig1)
            plt.close(fig20)
            plt.close(fig10)
            return
        else:
            if not showE:
                plt.close(fig1)
                plt.close(fig10)
            if not showTau:
                plt.close(fig2)
                plt.close(fig20)
            return



# 3. Comparison and correlation between growth and mechanicas parameters

def GOC_Comp(GD_Growths,GD_OCs,ParamGrowth,ParamOC,labelsGrowth,labelsOC,Titles,colors, **kwargs):
    
    PlotFits = False
    CorrType = 'pearson'
    
    for key,value in kwargs.items():
        if key == 'PlotFits':
            PlotFits= value
        elif key == 'CorrType':
            CorrType = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    fullData = pd.DataFrame(data=None, columns=ParamOC+ParamGrowth)
    GDs = []
    
    for (GD_Growth,GD_OC,lab) in zip(GD_Growths,GD_OCs,Titles) :
        
        ListGrowth = np.unique(GD_Growth.index)
        ListChoc = np.unique(GD_OC.index)
        CommonList = set.intersection(set(ListGrowth), set(ListChoc))

        print('n = ' + str(len(CommonList)))

        DataFit = GD_OC.loc[(GD_OC['Img']==0),['Img']+ParamOC].loc[CommonList]
        DataGrowth = GD_Growth.loc[(GD_Growth['Img']==0),ParamGrowth].loc[CommonList]
        Data = DataFit.join(DataGrowth) 
        
        Data['Expe'] = lab
        
        fullData = fullData.append(Data, ignore_index=True)
        
        GDs.append(Data.copy(deep=True))
    
    GDs.insert(0,fullData.copy(deep=True))
    
    columns = ParamGrowth+ParamOC
    
    Corr(GDs,['Pooled'] + Titles,columns = columns,columnslabels = labelsGrowth+labelsOC,PlotFits = PlotFits,colors=colors, corrmethod =CorrType)
   
    
#%% Variability in size 

def sizeVar(GDs,labels,label,colors,**kwargs):
        
    showcurve = True
    showbox = False
    pooledGraph=True
    
    for key, value in kwargs.items(): 
        if key == 'showcurve':
            showcurve = value 
        elif key == 'pooledGraph':
            pooledGraph = value
        elif key == 'showbox':
            showbox = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    if showcurve:
        if pooledGraph:
            f0, ax0 = plt.subplots(ncols =2,dpi=300,figsize=(10,5))            
            f0.suptitle(label)            
            
            f1, ax1 = plt.subplots(ncols =2,dpi=300,figsize=(10,5))            
            f1.suptitle(label)
            
    
    for GD,lab,col in zip(GDs,labels,colors):
    
        nimgmax = GD['Img'].max() # number of images (duration) to plot for growth curve
    
    
        # number of ppgs and label
        nPPG = len(GD.loc[GD['Img'] == 0])
        lab = lab + ' - n = ' + str(nPPG)
        
        # Computing mean area over all gemmae for each image
        MeanA = np.empty(nimgmax)
        Time = np.empty(nimgmax)
        AadA = np.empty(nimgmax)
    
        for im in range(nimgmax):
    
            MeanA[im] = GD.loc[GD['Img'] == im,'Area'].to_numpy().mean()
            Time[im] = im/2
            AadA[im] = np.mean(np.abs(GD.loc[GD['Img'] == im,'Area'].to_numpy()-MeanA[im]))
        
        RAadA = np.divide(AadA,MeanA)*100
        
        if showcurve:
            
            if not pooledGraph:
                f0, ax0 = plt.subplots(ncols =2,dpi=300,figsize=(10,5))                
                f0.suptitle(label + ': ' + lab)        
                
                f1, ax1 = plt.subplots(ncols =2,dpi=300,figsize=(10,5))            
                f1.suptitle(label)
            
            ax0[0].set_title('Varibility in size over time')
            ax0[0].plot(Time,AadA, '-o', ms= 4, lw=1,color = col)
            ax0[0].set_xlabel('Time (hours)')
            ax0[0].set_ylabel('Variability in size (mm²)')
            
            ax0[1].set_title('Relative variability in size over time')
            ax0[1].plot(Time,RAadA, 'd-', ms = 4, lw=1, color = col)
            ax0[1].set_xlabel('Time (hours)')
            ax0[1].set_ylabel('Relative variability in size (%)')
            ax0[1].set_ylim(bottom=0,top=np.max([1.1*RAadA.max(),ax0[1].get_ylim()[1]]))
            
            
            
            ax1[0].set_title('Varibility in size vs mean size')
            ax1[0].plot(MeanA,AadA, 'o', ms= 4, lw=1,color = col)
            ax1[0].set_xlabel('Size (mm²)')
            ax1[0].set_ylabel('Variability in size (mm²)')
            
            ax1[1].set_title('Relative variability in size vs mean size')
            ax1[1].plot(MeanA,RAadA, 'd', ms = 4, lw=1, color = col)
            ax1[1].set_xlabel('Size (mm²)')
            ax1[1].set_ylabel('Relative variability in size (%)')
            ax1[1].set_ylim(bottom=0,top=np.max([1.1*RAadA.max(),ax0[1].get_ylim()[1]]))
            
            if not pooledGraph:
                f0.tight_layout
                f1.tight_layout
                
    if showcurve:        
        if pooledGraph:
            ax0[0].legend(labels)
            ax0[1].legend(labels)
            f0.tight_layout
            
            ax1[0].legend(labels)
            ax1[1].legend(labels)
            f1.tight_layout
            

#%% Growth rate ratios after/ before OC

def plotGRratio(GDs,GD_Osmos,labels,colors):
    AllRatios = np.empty(0)
    AllGRvar = np.empty(0)
    
    n = len(GDs)
    
    # Figure for GR ratios around OC pooled between experiments
    f0,axes = plt.subplot_mosaic(vf.mosaicList(n)[0], dpi=200, figsize=(5,5))
    f0.patch.set_facecolor('white')
    axes['a'].set_title('Growth rates change\n caused by Osmotic choc')
    
    # Figure for GR ratios during growth pooled between experiments
    f1,ax1 = plt.subplots(dpi=200)
    f1.patch.set_facecolor('white')
    f1.suptitle('All growth GR pooled')
    
    
    # Figure for GR ratios of both cases
    f01,ax01 = plt.subplots(dpi=200)
    f01.patch.set_facecolor('white')
    f01.suptitle('GR ratios comparison')
    
    
    for GD,GD_Osmo,label,colo,nax in zip(GDs,GD_Osmos,labels,colors,vf.mosaicList(n)[1]):
        
        # GR ratio before/after OC
        GRbefore = GD.loc[GD['Img']==0,'GR_end']
        GRafter = GD_Osmo.loc[GD_Osmo['Img']==0,'GR_AfterOC']
        GRs = pd.concat([GRbefore, GRafter],axis=1)
        GRs = GRs.assign(GR_ratio = lambda x: (x['GR_AfterOC'] / x['GR_end']))
        
        Ratios = GRs['GR_ratio'].to_numpy()
        Ratios = Ratios[~np.isnan(Ratios)]
        Ratios_Valid = Ratios[(Ratios>0)&(Ratios<4)]
        
        AllRatios = np.append(AllRatios,Ratios_Valid)
        
        
        # GR ratio during growth
        GRvar_Exp = np.empty(0)
        GRvarMedians = np.empty(0)    
        
        for s in np.unique(GD.index):
            GR = GD.loc[s,'GR_Full'].to_numpy()
        
            GRvar = np.abs(np.divide(GR[1:-1],GR[0:-2]))

            
            GRvar_Exp = np.append(GRvar_Exp,GRvar)
            GRvarMedians = np.append(GRvarMedians,np.median(GRvar))
            
            # GRvarInWindow = np.round(len(GRvar[(GRvar>0) & (GRvar<2)])/len(GRvar)*1000)/10
            # f,ax = plt.subplots(dpi=200)
            # f.patch.set_facecolor('white')
            # f.suptitle(s + ' (' + str(GRvarInWindow) + '% in window) + median')
            # ax.hist(GRvar, range = (0,2), bins = 21,color='r', density = True)
            # lims = ax0.get_ylim()
            # ax.plot([np.median(GRvar),np.median(GRvar)],lims,'--',color = 'lightgray')
            # ax.set_ylim(lims)
        
        AllGRvar = np.append(AllGRvar,GRvar_Exp)
        
        # Plot        
        # axes[nax].hist(GRvar_Exp, range = (0,2), bins = 15,color='gray', density = True, label = 'GR ratios during growth')
        axes[nax].hist(GRs['GR_ratio'], range = (0,4), bins = 12, density = True, color = colo, alpha = 0.7)
        
        # axes[nax].set_xlabel('GR ratios')
        axes[nax].set_xlim([-1,4])
        
        sns.kdeplot(Ratios_Valid,ax=axes[nax], color = 'k',lw=1)
        
        sns.kdeplot(Ratios_Valid,ax=axes['a'],color = colo, label= label)
        axes[nax].set_ylabel('')


    ax1.hist(GRvar_Exp, range = (0,2), bins = 15,color='b', density = True, label = 'GR ratios during growth')
    ax1.set_ylabel('Density')
    ax1.set_xlabel('GR ratios')

    axes['a'].hist(AllRatios, range = (0,4), bins = 8,color='gray', density = True, label = 'Pooled data')
    axes['a'].set_ylabel('Density')
    axes['a'].set_xlabel('Growth Rate ratios after/before Osmotic choc')
    axes['a'].set_xlim([-1,4.5])
    # axes['a'].set_title('Mean : ' + '{0:.2f}'.format(AllRatios.mean()))
    axes['a'].legend()
    f0.tight_layout()
    
    ax01.hist(AllRatios, range = (0,4), bins = 8,color='gray', density = True, label = 'GR ratios around OC', alpha = 0.7)
    ax01.hist(AllGRvar, range = (0,2), bins = 12,color='b', density = True, label = 'GR ratios during growth', alpha = 0.7)
    ax01.set_ylabel('Density')
    ax01.set_xlabel('GR ratios')
    ax01.set_xlim([-1,4])
    ax01.legend()
    f01.tight_layout()
   
   
#%% Comparison of successive OC on the same gemmae population

def successiveOC(GD1,GD2):
    
    GD2.loc[:,'Ecomp2'] = GD2.loc[:,'Ecomp'] 
    GD2.loc[:,'Erel2'] = GD2.loc[:,'Erel'] 
    GD2 = GD2.drop(columns=['Ecomp','Erel'])
    
    
    GD = pd.merge(GD1.loc[GD1['Img']==0,:],GD2.loc[GD2['Img']==0,:], left_index=True, right_index=True)
    
    Eoc1 = GD.loc[:,'Ecomp'].to_numpy()
    Eoc2 = GD.loc[:,'Ecomp2'].to_numpy()
    
    ncommon = len(Eoc1)
    ratio21 = np.divide(Eoc2,Eoc1)
    
    f, ax = plt.subplots(dpi=300)
    f.suptitle('Ratio of E, second choc/first choc')
    ax.hist(ratio21,density =True)
    ax.set_ylabel('Density')
    ax.set_xlabel('E2/E1')
    ax.set_title('n = ' + str(ncommon) + '. Mean = ' + str(np.round(np.mean(ratio21)*100)/100))
    
    
#%% Plots of rhizoides appearance time
    
def plotRhizoides(Ps,colors,names,labels,nimgmaxes):
    
    f, ax = plt.subplots(dpi=200)
    f1, ax1 = plt.subplots(dpi=200)
    f2, ax2 = plt.subplots(dpi=200)
    
    for P,col,name,lab,ci,nimgmax in zip(Ps,colors,names,labels,range(len(Ps)),nimgmaxes):
        
        RD = pd.read_excel(P + '\\RhizoideData' + name + '.xlsx',index_col = 'Ind')
        
        List = np.unique(RD.index) 
        
        RhizExit = RD[['RhizExit']].to_numpy().astype('float')  
        
        RhizExitAl = RD[['RhizExitAl']].to_numpy().astype('float')    
        
        RhizOut = RhizExit<200
        
        RhizExit = RhizExit[RhizOut]
        RhizExitAl = RhizExitAl[RhizOut]
        
        time = np.arange(nimgmax)/2
    
        plotprops = {'color':'black'}
        boxprops = {'color':'black','facecolor':col}
        ax.boxplot(RhizExit/2, positions = [ci], labels = [lab],patch_artist = True, 
                   boxprops=boxprops, capprops =plotprops,showfliers = False,whiskerprops=plotprops,medianprops =plotprops) 
        
        
        RhizFrac = np.empty(nimgmax)
        RhizFracAl = np.empty(nimgmax)
        for ii in range(nimgmax):
            RhizFrac[ii] = np.sum(RhizExit<ii+1)/len(List)*100
            RhizFracAl[ii] = np.sum(RhizExitAl<ii+1)/len(List)*100
            
        ax1.plot(time,RhizFrac,'-o',color = col)
        ax2.plot(time,RhizFracAl,'-o',color = col)
        
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    
    ax.set_ylabel('Time of first rhizoides (hours)')
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('% of gemmae with visible rhizoides')
    ax1.legend(handles, labels)
    
    ax2.set_xlabel('Time after growth start (hours)')
    ax2.set_ylabel('% of gemmae with visible rhizoides')
    ax2.legend(handles, labels)
    


#%% Plots for shape analysis

# 1. Distances to mean contours

def plotMeanDist(GDs,Expname,colors,Times):
    
        
    ## Evolution in time per experiment
    
    for GD,col,j in zip(GDs,colors,range(len(Expname))):
        
 

    
        fullMedDist = []
        fullMedDistN = []
        # fullL2Dist = []
        # fullL2DistN = []
        
        ## Main code
        for t in Times:
            
            MedDist = GD.loc[(GD['MeanCimg']==t*2),'MedDist']
            MedDistN = GD.loc[(GD['MeanCimg']==t*2),'MedDistNorm']
            # L2Dist = GD.loc[(GD['MeanCimg']==t*2),'L2Dist']
            # L2DistN = GD.loc[(GD['MeanCimg']==t*2),'L2DistNorm']
            
            fullMedDist.append(MedDist)
            fullMedDistN.append(MedDistN)
            # fullL2Dist.append(L2Dist)
            # fullL2DistN.append(L2DistN)
            
            
            

        cols = [col for i in Times]

        fig0, ax0, _, _ = vf.boxswarmplot('Evolution of median distance to mean for ' + Expname[j],'Distance (µm)',
                     fullMedDist,cols,[str(t) + 'h' for t in Times], showN = False)
        ax0.set_xlabel('Time after dormancy exit (hours)')
        
        
        fig1, ax1, _, _ = vf.boxswarmplot('Evolution of median distance to normalized mean for ' + Expname[j],'Distance (norm)',
                     fullMedDistN,cols,[str(t) + 'h' for t in Times], showN = False)
        ax1.set_xlabel('Time after dormancy exit (hours)')
        
        
        # fig2, ax2, _, _ = vf.boxswarmplot('Evolution of L2 distance to mean for ' + Expname[j],'L2 distance to mean contour (µm)',
        #              fullL2Dist,cols,[str(t) + 'h' for t in Times], showN = False)
        # ax2.set_xlabel('Time after dormancy exit (hours)')
        
        
        # fig3, ax3, _, _ = vf.boxswarmplot('Evolution of L2 distance to normalized mean for ' + Expname[j],'L2 distance to mean contour (norm)',
        #              fullL2DistN,cols,[str(t) + 'h' for t in Times], showN = False)
        # ax3.set_xlabel('Time after dormancy exit (hours)')
        
        
    
        
     ## Comparison between experiments    
    if len(GDs)>1:       
        for t in Times:

            # MedDist = [None]*len(GDs)
            MedDistN = [None]*len(GDs)

            for j,GD in enumerate(GDs):

                # MedDist[j] = GD.loc[(GD['MeanCimg']==t*2),'MedDist']
                MedDistN[j] = GD.loc[(GD['MeanCimg']==t*2),'MedDistNorm']


            # fig10, ax10, cap, _ = vf.boxswarmplot('Distance to mean ' + str(t) + ' hours after dormancy exit',
            #                                     'Distance (µm)',
            #              MedDist,colors,Expname[:], showN = True)


            fig11, ax11, capN, _ = vf.boxswarmplot('Distance to normalized mean ' + str(t) + ' hours after dormancy exit',
                                                'Distance (norm)',
                         MedDistN,colors,Expname[:], showN = True)


            # step = np.max(cap)*0.125
            stepN = np.max(capN)*0.125
            # fullstep = 0
            fullstepN = 0
            # hmax = np.max(cap)
            hmaxN = np.max(capN)


            for i in range(len(GDs)-1):
                for j in range(i+1,len(GDs)):

                    # fullstep = plotSig(ax10,hmax,step,fullstep,MedDist[i],MedDist[j],i,j)
                    fullstepN = plotSig(ax11,hmaxN,stepN,fullstepN,MedDistN[i],MedDistN[j],i,j)
                    
                    
                    
 # 2. Growth
 
def plotGrowth(PFig,GrowthD,CD,Expname,DPI):
    
    StackList = np.unique(GrowthD.index)
    
    Pfolder = PFig + '\\ContoursGrowth\\' 
    
    if not os.path.exists(Pfolder):
        os.mkdir(Pfolder) # create general folder  
    
    Pfolder = Pfolder +  Expname + '\\'
    
    if not os.path.exists(Pfolder):
        os.mkdir(Pfolder) # create general folder  
        
    Pinstant = Pfolder + 'InstantGrowth\\'
    
    if not os.path.exists(Pinstant):
        os.mkdir(Pinstant) # create folder  
    
    Paccum = Pfolder + 'AccumulatedGrowth\\'
    
    if not os.path.exists(Paccum):
        os.mkdir(Paccum) # create folder  
    
    Ptot = Pfolder + 'TotalGrowth\\'
    
    if not os.path.exists(Ptot):
        os.mkdir(Ptot) # createfolder  
        
    for s in StackList:
        
            
        fullP0 = Pinstant + s + '\\'
    
        fullP1 = Paccum + s + '\\'

        fullP2 = Ptot + s + '\\'
            
        if not os.path.exists(fullP2 + 'FullTotGrowth.png'):
        
            print('Plotting for : ' + s)
            
            nimg = np.max(GrowthD.loc[s,'Img'])
    
            if not os.path.exists(fullP0):
                    os.mkdir(fullP0) # create experiment folde
    
            if not os.path.exists(fullP1):
                    os.mkdir(fullP1) # create experiment folde
    
            if not os.path.exists(fullP2):
                    os.mkdir(fullP2) # create experiment folde
            
            minInstGrowth = GrowthD.loc[s,'InstantGrowth'].min()
            maxInstGrowth = np.percentile(GrowthD.loc[s,'InstantGrowth'].values,95)
            
            minAccGrowth = GrowthD.loc[s,'AccumulatedGrowth'].min()
            maxAccGrowth = GrowthD.loc[s,'AccumulatedGrowth'].max()
            
            minTotGrowth = GrowthD.loc[s,'TotGrowth'].min()
            maxTotGrowth = np.percentile(GrowthD.loc[s,'TotGrowth'].values,75)
            
            fig00,ax00 = plt.subplots(dpi = DPI)
            fig00.suptitle('Instant growth of ' + s)
            
            fig01,ax01 = plt.subplots(dpi = DPI)
            fig01.suptitle('Accumulated growth of ' + s)
            
            fig02,ax02 = plt.subplots(dpi = DPI)
            fig02.suptitle('Total growth of ' + s)
            
            for i in range(1,nimg):
                
                X = CD.loc[(CD.index == s) & (CD['Img'] == i),'X']
                Y = CD.loc[(CD.index == s) & (CD['Img'] == i),'Y']
                
                InstG = GrowthD.loc[(GrowthD.index == s) & (GrowthD['Img'] == i),'InstantGrowth']
                AccuG = GrowthD.loc[(GrowthD.index == s) & (GrowthD['Img'] == i),'AccumulatedGrowth']
                TotG = GrowthD.loc[(GrowthD.index == s) & (GrowthD['Img'] == i),'TotGrowth']
                
                fig0,ax0 = plt.subplots(dpi = DPI)
                fig0.suptitle('Growth between image ' + str(i-1) + ' & ' + str(i) + '.')            
                pts0 = ax0.scatter(X,Y,c = InstG, cmap = 'gist_rainbow_r',s = 0.3,marker = 'o',vmax=maxInstGrowth,vmin=minInstGrowth)
                divider0 = make_axes_locatable(ax0)
                cax0 = divider0.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(pts0,cax = cax0)
                ax0.set_aspect('equal')
                fig0.savefig(fullP0 + 'InstGrowthImg' + str(i-1) + '_' + str(i))
                plt.close(fig0)
                
                
                fig1,ax1 = plt.subplots(dpi = DPI)
                fig1.suptitle('Accumulated growth a image ' + str(i) + '.')            
                pts1 = ax1.scatter(X,Y,c = AccuG, cmap = 'gist_rainbow_r',s = 0.3,marker = 'o',vmax=maxAccGrowth,vmin=minAccGrowth)
                divider1 = make_axes_locatable(ax1)
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(pts1,cax = cax1)
                ax1.set_aspect('equal')
                fig1.savefig(fullP1 + 'AccGrowthImg' + str(i))
                plt.close(fig1)
                
                
                fig2,ax2 = plt.subplots(dpi = DPI)
                fig2.suptitle('Total growth a image ' + str(i) + '.')          
                pts2 = ax2.scatter(X,Y,c = TotG, cmap = 'gist_rainbow_r',s = 0.3,marker = 'o',vmax=maxTotGrowth,vmin=minTotGrowth)
                divider2 = make_axes_locatable(ax2)
                cax2 = divider2.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(pts2,cax = cax2)
                ax2.set_aspect('equal')
                fig2.savefig(fullP2 + 'TotGrowthImg' + str(i))
                plt.close(fig2)
                
                pts00 = ax00.scatter(X,Y,c = InstG, cmap = 'gist_rainbow_r',s = 0.2,marker = 'o',vmax=maxInstGrowth,vmin=minInstGrowth)
                pts01 = ax01.scatter(X,Y,c = AccuG, cmap = 'gist_rainbow_r',s = 0.2,marker = 'o',vmax=maxAccGrowth,vmin=minAccGrowth)
                pts02 = ax02.scatter(X,Y,c = TotG, cmap = 'gist_rainbow_r',s = 0.2,marker = 'o',vmax=maxTotGrowth,vmin=minTotGrowth)
                
                del X, Y, InstG, AccuG, TotG
                
                gc.collect() 
    
            divider00 = make_axes_locatable(ax00)
            cax00 = divider00.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(pts00,cax = cax00)
            ax00.set_aspect('equal') 
            fig00.savefig(fullP0 + 'FullInstantGrowth')
            plt.close(fig00)
                             
            divider01 = make_axes_locatable(ax01)
            cax01 = divider01.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(pts01,cax = cax01)
            ax01.set_aspect('equal') 
            fig01.savefig(fullP1 + 'FullAccumulatedGrowth')
            plt.close(fig01)
                             
            divider02 = make_axes_locatable(ax02)
            cax02 = divider02.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(pts02,cax = cax02)
            ax02.set_aspect('equal') 
            fig02.savefig(fullP2 + 'FullTotGrowth')
            plt.close(fig02)
                
            
            del minInstGrowth , maxInstGrowth, minAccGrowth, maxAccGrowth, minTotGrowth, maxTotGrowth
            
            gc.collect() 
            
        else:
            
            print(s + ' is already plotted.')
            
            
            
