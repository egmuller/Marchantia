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


import numpy as np
import pandas as pd

import os
#%% Data visualisation function for growth and osmotic chocs quantifications

# 1. Comparison of growth quantifications between different experiments

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
    
    fig2,ax2 = plt.subplots(dpi = 250,facecolor='white')
    fig2.suptitle(Title + ' - Area vs. time')
    plt.xlabel('Time (min)')
    plt.ylabel('Area (mm²)')
    
    fig3,ax3 = plt.subplots(dpi = 250,facecolor='white')
    fig3.suptitle(Title + ' - Norm Area vs. time')
    plt.xlabel('Time (min)')
    plt.ylabel('Area (normalized)')

    
    for GD,lab,i in zip(newGDs,Labels,range(len(GDs))):
        
        StackList = np.unique(GD.index)
           
        if NimgMax == 'max':
            nimgmax = GD['Img'].max() # number of images (duration) to plot for growth curve
        else:
            nimgmax = 49 # 24h
        
        if IndividualPlots:
            fig1,ax1 = plt.subplots(dpi = 250,facecolor='white')
            fig1.suptitle(lab + ' - Area vs. time')
            plt.xlabel('Time (min)')
            plt.ylabel('Area (mm²)')
            for s in StackList:
                ax1.plot(GD.loc[s,'Img']*30,GD.loc[s,'Area'],label=s,lw=1)
            plt.legend(prop={'size': 5})

        # number of ppgs and label
        nPPG = len(GD.loc[GD['Img'] == 0])
        lab = lab + ' - n = ' + str(nPPG)
        
        # Computing mean area over all gemmae for each image
        MeanA = np.empty(nimgmax)
        MeanTime = np.empty(nimgmax)
        StdA = np.empty(nimgmax)

        for im in range(nimgmax):

            MeanA[im] = GD.loc[GD['Img'] == im,'Area'].to_numpy().mean()
            MeanTime[im] = im*30
            StdA[im] = GD.loc[GD['Img'] == im,'Area'].to_numpy().std()
        
        nppg = len(GD.loc[GD['Img'] == 0,'Area'].to_numpy())

        ax2.errorbar(MeanTime,MeanA,yerr=StdA/np.sqrt(nppg), capsize=3,label=lab,color = colors[i])
        ax3.errorbar(MeanTime,MeanA/MeanA[0],yerr=StdA/MeanA[0]/np.sqrt(nppg), capsize=3,label=lab,color = colors[i])
        
    plt.figure(fig2.number)
    plt.legend(prop={'size': 8})
    fig2.savefig(P + '\\AreaGrowth\\' + Title + '_AreaCurve.png')
    if not showcurve:
        plt.close(fig2)

    plt.figure(fig3.number)
    plt.legend(prop={'size': 8})
    fig3.savefig(P + '\\AreaGrowth\\' + Title + '_NormAreaCurve.png')
    if not showcurve:
        plt.close(fig3)


    ######### Parameters of fit ###########
    
      
    fig4,ax4 = plt.subplots(dpi = 250,facecolor='white')
    fig4.suptitle(Title + ' - Growth start time')
    plt.ylabel('Tstart (hours)')
      
    fig5,ax5 = plt.subplots(dpi = 250,facecolor='white')
    fig5.suptitle(Title + ' - Growth caracteristic time')
    plt.ylabel('Tau growth (hours)')
    
    fig6,ax6 = plt.subplots(dpi = 250,facecolor='white') 
    fig6.suptitle(Title + ' - Starting area') 
    plt.ylabel('Starting area from fit (mm²)') 

    fig16,ax16 = plt.subplots(dpi = 250,facecolor='white')
    fig16.suptitle(Title + ' - Initial growth increase')
    plt.ylabel('Growth at Tstart (%)')
    
    if len(newGDs) == 2:
        # Histogram for distribution comparison
        fig7,ax7 = plt.subplots(dpi = 250,figsize = (5,3.5),facecolor='white')
        fig7.suptitle(Title + ' - Growth caracteristic times')
        plt.xlabel('Tau growth (hours)')
        plt.ylabel('PDF')

        fig8,ax8 = plt.subplots(dpi = 250,figsize = (5,3.5),facecolor='white')
        fig8.suptitle(Title + ' - Growth start time')
        plt.xlabel('Tstart (min)')
        plt.ylabel('PDF')
        
        fig9,ax9 = plt.subplots(dpi = 250,figsize = (5,3.5),facecolor='white')
        fig9.suptitle(Title + ' - Growth caracteristic times')
        plt.xlabel('Tau growth (hours) - median aligned')
        plt.ylabel('PDF')

        fig10,ax10 = plt.subplots(dpi = 250,figsize = (5,3.5),facecolor='white')
        fig10.suptitle(Title + ' - Growth start time')
        plt.xlabel('T start (min) - median aligned')
        plt.ylabel('PDF')
    
    tdebs= [None]*len(newGDs)
    taus= [None]*len(newGDs)
    captdeb= [None]*len(newGDs)
    captau= [None]*len(newGDs)
    medtdeb= [None]*len(newGDs)
    medtau= [None]*len(newGDs)    
    
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
        taus[i] = GD.loc[GD['Img'] == 0, 'Tau']/60 *np.log(2)         
        Area0[i] = GD.loc[GD['Img'] == 0, 'A0fit'] 
        AreaStart[i] = GD.loc[GD['Img'] == 0, 'GrowthAtStart_flat']*100
        
        
        # swarmplots
        grouping = np.append(grouping,np.ones(len(tdebs[i]))*i)


        plotprops = {'color':'black'}
        boxprops = {'color':'black','facecolor':colors[i]}
        
         
        bp4 = ax4.boxplot(tdebs[i], positions = [i], labels = [lab],patch_artist = True, boxprops=boxprops, capprops =plotprops,
                    showfliers = False,whiskerprops=plotprops,medianprops =plotprops)
        
        bp5 = ax5.boxplot(taus[i], positions = [i], labels = [lab],patch_artist = True, boxprops=boxprops, capprops =plotprops,
                    showfliers = False,whiskerprops=plotprops,medianprops =plotprops)
        
        bp6 = ax6.boxplot(Area0[i], positions = [i], labels = [lab],patch_artist = True, boxprops=boxprops, capprops =plotprops, 
            showfliers = False,whiskerprops=plotprops,medianprops =plotprops) 
        
        bp26 = ax16.boxplot(AreaStart[i], positions = [i], labels = [lab],patch_artist = True, boxprops=boxprops, capprops =plotprops,
                    showfliers = False,whiskerprops=plotprops,medianprops =plotprops)
    
        captdeb[i] = bp4['caps'][1].get_ydata(orig=True)[0]
        captau[i] = bp5['caps'][1].get_ydata(orig=True)[0]
        capArea0[i] = bp6['caps'][1].get_ydata(orig=True)[0] 
        capAreaStart[i] = bp26['caps'][1].get_ydata(orig=True)[0]
        medtdeb[i] = bp4['medians'][0].get_ydata(orig=True)[0]
        medtau[i] = bp5['medians'][0].get_ydata(orig=True)[0]
        medArea0[i] = bp6['medians'][0].get_ydata(orig=True)[0] 
        medAreaStart[i] = bp26['medians'][0].get_ydata(orig=True)[0]
        
        if len(newGDs) == 2:
        
            ax7.hist(taus[i], nbins, density=True, facecolor=colors[i], alpha=0.5)
            ax8.hist(tdebs[i], nbins, density=True, facecolor=colors[i], alpha=0.5)
            ax9.hist(taus[i]-np.median(taus[i]), nbins, density=True, facecolor=colors[i], alpha=0.5)
            ax10.hist(tdebs[i]-np.median(tdebs[i]), nbins, density=True, facecolor=colors[i], alpha=0.5)

            
    sns.swarmplot(x=grouping,y=pd.concat(tdebs),color = 'gray', size=2, ax = ax4)
    sns.swarmplot(x=grouping,y=pd.concat(taus),color = 'gray', size=2, ax = ax5)
    sns.swarmplot(x=grouping,y=pd.concat(Area0),color = 'gray', size=2, ax = ax6) 
    sns.swarmplot(x=grouping,y=pd.concat(AreaStart),color = 'gray', size=2, ax = ax16)
    
    ax4.set_xticklabels(labs)
    ax5.set_xticklabels(labs)
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
    stepArea0 = np.max(capArea0)*0.125 
    stepAreaStart = np.max(capAreaStart)*0.125
    
    fullsteptdeb = 0
    fullsteptau = 0
    fullstepArea0 = 0 
    fullstepAreaStart = 0
    
    hmaxtdeb = np.max(captdeb)
    hmaxtau = np.max(captau)
    hmaxArea0 = np.max(capArea0) 
    hmaxAreaStart = np.max(capAreaStart)
    
    if stats=='ranksum':
        if AllSigs:
            for i in range(len(newGDs)-1):
                for j in range(i+1,len(newGDs)):

                    fullsteptdeb = plotSig(ax4,hmaxtdeb,steptdeb,fullsteptdeb,tdebs[i],tdebs[j],i,j)

                    fullsteptau = plotSig(ax5,hmaxtau,steptau,fullsteptau,taus[i],taus[j],i,j)
                    
                    fullstepArea0 = plotSig(ax6,hmaxArea0,stepArea0,fullstepArea0,Area0[i],Area0[j],i,j) 

                    fullstepAreaStart = plotSig(ax16,hmaxAreaStart,stepAreaStart,fullstepAreaStart,AreaStart[i],AreaStart[j],i,j)
                    
        else:
            for i,j in sigpairs:

                fullsteptdeb = plotSig(ax4,hmaxtdeb,steptdeb,fullsteptdeb,tdebs[i],tdebs[j],i,j)

                fullsteptau = plotSig(ax5,hmaxtau,steptau,fullsteptau,taus[i],taus[j],i,j)
 
                fullstepArea0 = plotSig(ax6,hmaxArea0,stepArea0,fullstepArea0,Area0[i],Area0[j],i,j) 

                fullstepAreaStart = plotSig(ax16,hmaxAreaStart,stepAreaStart,fullstepAreaStart,AreaStart[i],AreaStart[j],i,j)

    elif stats == 'kruskal':
        
        StatsKruskal(ax4,tdebs)
        StatsKruskal(ax5,taus)
        StatsKruskal(ax6,Area0)
        StatsKruskal(ax16,AreaStart)
               
    ax4.set_ylabel('Tstart (hours)')
      
    ax5.set_ylabel('Area doubling (hours)')
    
    ax6.set_ylabel('Starting area from fit (mm²)') 

    ax16.set_ylabel('Growth at Tstart (%)')   
 
    if stats=='ranksum':
        fig4.savefig(P + '\\AreaGrowth\\' + Title + '_Tstart.png')
        fig5.savefig(P + '\\AreaGrowth\\'+ Title +  '_TauGrowth.png')
        fig6.savefig(P + '\\AreaGrowth\\'+ Title +  '_StartingArea.png') 
        fig16.savefig(P + '\\AreaGrowth\\'+ Title +  '_InitialGrowth.png')
        if not showbox:
            plt.close(fig5)
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
    Es= [None]*len(GDs)
    Ecomps= [None]*len(GDs)
    Lcomps= [None]*len(GDs)
    Erels= [None]*len(GDs)
    Lrels= [None]*len(GDs)
    
    AllRatios = np.empty(0)
    
    n = len(GDs)
    
    # Figure for E ratios 
    f3,ax3 = plt.subplot_mosaic(vf.mosaicList(n)[0], dpi=200, figsize=(7,5))
    f3.patch.set_facecolor('white')
    ax3['a'].set_title('Growth rates change\n caused by Osmotic choc')
    
    
    
    for GD,lab,i,nax in zip(GDs,Labels,range(len(GDs)),vf.mosaicList(n)[1]):
        
        # Retrieve data
        Es[i] = GD.loc[GD['Img'] == 0, 'E']
        Ecomps[i] = GD.loc[GD['Img'] == 0, 'Ecomp']
        Lcomps[i] = GD.loc[GD['Img'] == 0, 'TauFlux'] 
        Erels[i] = GD.loc[GD['Img'] == 0, 'Erel']
        Lrels[i] = GD.loc[GD['Img'] == 0, 'TauFluxRel']  
        
        
        Eratios = np.divide(Erels[i],Ecomps[i])
        AllRatios = np.append(AllRatios,Eratios)
        
        if indiplots:
            fig0,ax0,cap,med = vf.boxswarmplot(Title + '\n\nElastic bulk modulus comparison for ' + lab,'E (MPa)',
                                               [Ecomps[i],Erels[i]],[colors[i],colors[i]],['Ec','Ei'])
    
            plotSig(ax0,np.max(cap),np.max(cap)*0.125,0,Ecomps[i],Erels[i],0,1)
            
            fig0.tight_layout() 
            fig0.savefig(P + '\\Hydromechanics\\' + lab + '_EComp-Rel.png')
            if not showE:
                plt.close(fig0)
            
            fig01,ax01,cap,med = vf.boxswarmplot(Title + '\n\nTauFlux comparison for ' + lab,'Tau (min-1)',
                                               [Lcomps[i],Lrels[i]],[colors[i],colors[i]],['TauFlux_comp','TauFlux_rel'])
    
            plotSig(ax01,np.max(cap),np.max(cap)*0.125,0,Lcomps[i],Lrels[i],0,1)
            
            fig01.tight_layout()
            fig01.savefig(P + '\\Hydromechanics\\' + lab + '_Tflux-Rel.png')
            if not showTau:
                plt.close(fig01)
            
            
            if showhist:
                fig00, ax00 = plt.subplots(dpi=300)
                ax00.hist(Eratios, facecolor=colors[i]) # ,density = True
                fig00.suptitle('Median : ' + str(np.round(Eratios.median()*100)/100) + 
                               ' - Mean : ' + str(np.round(Eratios.mean()*100)/100))
                ax00.set_xlabel('Ei/Ec')
                ax00.set_ylabel('Count')
                fig00.savefig(P + '\\Hydromechanics\\' + lab + '_EComp-Rel_Dist.png')
                if not showE:
                    plt.close(fig00)
                
                
                fig00, ax00 = plt.subplots(dpi=300)
                ax00.hist(Es[i], facecolor=colors[i],bins=20) # ,density = True
                fig00.suptitle('Bulk elastic moduli')
                ax00.set_xlabel('E (MPa)')
                ax00.set_ylabel('Count')
                # ax00.set_xlim(right=1.5)
                fig00.savefig(P + '\\Hydromechanics\\' + lab + '_E_Dist.png')
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
                ax001.hist(np.divide(Lrels[i],Lcomps[i]), facecolor=colors[i]) # ,density = True
                fig001.suptitle('Median : ' + str(np.round(np.divide(Lrels[i],Lcomps[i]).median()*100)/100) + 
                               ' - Mean : ' + str(np.round(np.divide(Lrels[i],Lcomps[i]).mean()*100)/100))
                ax001.set_xlabel('TfluxRel/Tflux')
                ax001.set_ylabel('Count')
                fig001.savefig(P + '\\Hydromechanics\\' + lab + '_TauFluxComp-Rel_Dist.png')
                if not showTau:
                    plt.close(fig001)
                
        if showhist:   
            ax3[nax].hist(Eratios, facecolor=colors[i], density = True)
            sns.kdeplot(Eratios,ax=ax3[nax], color = 'k',lw=1)
             
            sns.kdeplot(Eratios,ax=ax3['a'],color = colors[i], label= lab)
            ax3[nax].set_ylabel('')
            ax3[nax].set_xlabel(lab)
    
    ### E ratios histograms
    ax3['a'].hist(AllRatios,color='gray', density = True, label = 'Pooled data')
    ax3['a'].set_ylabel('Density')
    ax3['a'].set_xlabel('Ei/Ec')
    ax3['a'].set_title('Mean : ' + '{0:.2f}'.format(AllRatios.mean()))
    ax3['a'].legend()
    f3.tight_layout()
    
    
    
    ### boxplots
    fig1,ax1,capEcomp,medEcomp = vf.boxswarmplot(Title + '\n\nElastic bulk modulus (compression)','Ec (MPa)',Ecomps,colors,Labels[:])
    fig10,ax10,capErel,medErel = vf.boxswarmplot(Title + '\n\nElastic bulk modulus (relaxation)','Ei (MPa)',Erels,colors,Labels[:])
    fig11,ax11,capE,medE = vf.boxswarmplot(Title + '\n\nElastic bulk modulus (mean)','E (MPa)',Es,colors,Labels[:])
    fig2,ax2,capLcomp,medLcomp = vf.boxswarmplot(Title + '\n\nTauFlux (compression)','Tau_Comp (min-1)',Lcomps,colors,Labels[:])
    fig20,ax20,capLrel,medLrel = vf.boxswarmplot(Title + '\n\nTauFlux (relaxation)','Tau_Rel (min-1)',Lrels,colors,Labels[:])       


    ### stats
    fullstepE = 0
    fullstepEcomp = 0
    fullstepLcomp = 0
    fullstepErel = 0
    fullstepLrel = 0
    
    if stats=='ranksum':
        if AllSigs:
            for i in range(len(GDs)-1):
                for j in range(i+1,len(GDs)):

                    fullstepEcomp = plotSig(ax1,np.max(capEcomp),np.max(capEcomp)*0.125,fullstepEcomp,Ecomps[i],Ecomps[j],i,j)
                    fullstepE = plotSig(ax11,np.max(capE),np.max(capE)*0.125,fullstepE,Es[i],Es[j],i,j)
                    fullstepErel = plotSig(ax10,np.max(capErel),np.max(capErel)*0.125,fullstepErel,Erels[i],Erels[j],i,j)
                    fullstepLcomp = plotSig(ax2,np.max(capLcomp),np.max(capLcomp)*0.125,fullstepLcomp,Lcomps[i],Lcomps[j],i,j)
                    fullstepLrel = plotSig(ax20,np.max(capLrel),np.max(capLrel)*0.125,fullstepLrel,Lrels[i],Lrels[j],i,j)

        else:
            for i,j in sigpairs:

                    fullstepEcomp = plotSig(ax1,np.max(capEcomp),np.max(capEcomp)*0.125,fullstepEcomp,Ecomps[i],Ecomps[j],i,j)
                    fullstepE = plotSig(ax11,np.max(capE),np.max(capE)*0.125,fullstepE,Es[i],Es[j],i,j)
                    fullstepErel = plotSig(ax10,np.max(capErel),np.max(capErel)*0.125,fullstepErel,Erels[i],Erels[j],i,j)
                    fullstepLcomp = plotSig(ax2,np.max(capLcomp),np.max(capLcomp)*0.125,fullstepLcomp,Lcomps[i],Lcomps[j],i,j)
                    fullstepLrel = plotSig(ax20,np.max(capLrel),np.max(capLrel)*0.125,fullstepLrel,Lrels[i],Lrels[j],i,j)

    fig1.tight_layout()
    fig2.tight_layout()
    fig10.tight_layout()
    fig20.tight_layout()
    fig11.tight_layout()
    

    if stats=='ranksum':
        fig1.savefig(P + '\\Hydromechanics\\' + Title + '_Ecomp.png')
        fig2.savefig(P + '\\Hydromechanics\\'+ Title +  '_TauFluxComp.png')
        fig10.savefig(P + '\\Hydromechanics\\' + Title + '_Erel.png')
        fig20.savefig(P + '\\Hydromechanics\\'+ Title +  '_TauFluxrel.png')
        fig11.savefig(P + '\\Hydromechanics\\' + Title + '_E.png')
        if not showbox:
            plt.close(fig2)
            plt.close(fig1)
            plt.close(fig20)
            plt.close(fig10)
            plt.close(fig11)
        else:
            if not showE:
                plt.close(fig1)
                plt.close(fig10)
                plt.close(fig11)
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
            plt.close(fig11)
            return
        else:
            if not showE:
                plt.close(fig1)
                plt.close(fig10)
                plt.close(fig11)
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
    
    
    
    
    
    