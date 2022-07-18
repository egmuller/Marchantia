# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:03:52 2022

@author: laplaud
"""


# Imports
import numpy as np
# import pandas as pd

from itertools import compress

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.stats import linregress

import VallapFunc as vf

import matplotlib.pyplot as plt



#%% Fit functions

# 1. Exponetial growth with a delay

def fitFunc(t,T,tdeb,A0): # Exponential growth with a delay
    
    f = np.multiply(A0,np.exp(np.divide((t-tdeb),T)))
        
    f[t<tdeb] = A0
        
    return(f)

# 1bis. Squared exponential follow by normal exponential to fit growth start correctly

def fitFuncMixed(t,T,L,A0): # Exponential growth with a delay
    
    f = np.multiply(A0,np.exp(np.divide(np.square(t),np.multiply(t,T)+L)))
    return(f)
        
def fitFuncMixed2(t,T,L,A0): # Exponential growth with a delay
    
    tdeb = L/(2*T) 
    
    f = np.multiply(A0,np.exp(np.divide((t-tdeb),T))) + np.multiply(A0,np.exp(np.divide(np.square(tdeb),L))-1)
    
    f[t<=tdeb] = np.multiply(A0,np.exp(np.divide(np.square(t[t<tdeb]),L)))

        
    return(f)

# 2. Exponential decay from initial volume to equilibrium volume

def fitFuncOsmChoc(t,T,A0,Aeq,tdeb): # 
    
    f = np.multiply((A0-Aeq),np.exp(np.divide(-(t-tdeb),T))) + Aeq
        
    f[t<tdeb] = A0
    
    return(f)

# 3. Exponential decay from initial volume to equilibrium volume + affine component after choc

def fitFuncOsmChoc2(t,T,A0,Aeq,tdeb,B): 
    
    f = np.multiply((A0-Aeq),np.exp(np.divide(-(t-tdeb),T))) + Aeq + np.multiply(t-tdeb,B)
        
    f[t<tdeb] = A0
    
    return(f)


#%% Growth curve fit

# Fit the evolution of area in time using a delayed exponential. A first fit 
# is used to determine the approximate value of growth start time. Then the
# fit is repeated only on data that go until 15h (FitWindow) after start time. 
# This is iterrated 3 times to have a convergence of the parameters.

# StackList : List of stack names to analyse, Rows : List of rows position 
# for each gemmae, GD : dataframe containing global (area) data, FPH : Frames per hour,
# Delay : time delay between gemmae imbibition and observation start

# Kwargs : 'debug' (True/False) for generating debug plots, 'fitwindow' (time in hours) 
# to choose the window after dormancy exit on which to fit 

def fitAreaGrowth(StackList,Rows,GD,FPH,Delay, **kwargs):
    
    DebugPlots = False
    FitWindow = 15
    
    for key, value in kwargs.items(): 
        if key == 'debug':
            DebugPlots = value
        if key == 'fitwindow':
            FitWindow = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')

    
    # Matrix for averaging growth rate
    
    GRmat = np.empty((200,len(StackList)))
    
    GRmat[:] = np.nan
            
    for ii,s,row in zip(range(len(StackList)),StackList,Rows):
        
        print('Fitting area curve for : ' + s.ljust(5), end='\n')           
        
        Time = GD.loc[s,'Img'].values.astype(float)/FPH*60 # in minutes
        AreaC = savgol_filter(GD.loc[s,'Area'].values, 11, 2)
        
        # first fit to determine tdeb
        params1, cov1 = curve_fit(f=fitFuncMixed2, xdata=Time, ydata=AreaC, p0=[100, 5000, AreaC[0]], bounds=(0, np.inf), method='trf', loss='soft_l1')
        stdevs1 = np.sqrt(np.diag(cov1))
        
        R2_1 = np.round(vf.computeR2(AreaC,fitFuncMixed2(Time,params1[0],params1[1],params1[2]))*1000)/1000
                        
        # Second fit, only until 15 hours after the start of growth
        fitInterval = Time<(params1[1]/(2*params1[0])+FitWindow*60)
        params2, cov2 = curve_fit(f=fitFuncMixed2, xdata=Time[fitInterval], ydata=AreaC[fitInterval], p0=params1,
                                  bounds=(0, np.inf), method='trf', loss='soft_l1')
        
        
        stdevs2 = np.sqrt(np.diag(cov2))
        
        R2_2 = np.round(vf.computeR2(AreaC[fitInterval],fitFuncMixed2(Time[fitInterval],params2[0],params2[1],params2[2]))*1000)/1000
        
        # Third fit, only until 15 hours after the start of growth, to confirm second
        fitInterval = Time<(params2[1]/(2*params2[0])+FitWindow*60)
        params3, cov3 = curve_fit(f=fitFuncMixed2, xdata=Time[fitInterval], ydata=AreaC[fitInterval], p0=params2,
                                  bounds=(0, np.inf), method='trf', loss='soft_l1')
        
        
        stdevs3 = np.sqrt(np.diag(cov2))
        
        R2_3 = np.round(vf.computeR2(AreaC[fitInterval],fitFuncMixed2(Time[fitInterval],params3[0],params3[1],params3[2]))*1000)/1000
        
        
        # Fourth fit, only until 15 hours after the start of growth, to confirm third
        fitInterval = Time<(params3[1]/(2*params3[0])+FitWindow*60)
        params4, cov4 = curve_fit(f=fitFuncMixed2, xdata=Time[fitInterval], ydata=AreaC[fitInterval], p0=params3,
                                  bounds=(0, np.inf), method='trf', loss='soft_l1')
        
        
        stdevs4 = np.sqrt(np.diag(cov2))
        
        R2_4 = np.round(vf.computeR2(AreaC[fitInterval],fitFuncMixed2(Time[fitInterval],params4[0],params4[1],params4[2]))*1000)/1000
        
        
        ### Growth rate 1/A * dA/dt computation
    
        dA = np.diff(AreaC)        
        dt = np.diff(Time)
        
        dAdt = np.divide(dA,dt)
        dAdt_S = savgol_filter(dAdt, 11, 3)
        

        
        intTime = Time[0:-1]+dt/2
        
        inv_A = np.divide(1,AreaC)
        inv_f = interp1d(Time,inv_A)
        inv_A_timed = inv_f(intTime)
        
        inv_A_S = savgol_filter(inv_A_timed,11, 3)
        
        
        GR = np.multiply(inv_A_timed,dAdt)
        GR_S = np.multiply(inv_A_S,dAdt_S)
        
        GR_2h = np.mean(GR_S[-4:])
        
        
        ### Computing growth start regime from growth rate
        
        r2 = 1
        Len = 2
        
        intTime_linfit = intTime[0:Len]
        GR_S_linfit = GR_S[0:Len]
        
        linreg = linregress(intTime_linfit,GR_S_linfit)
        
        Slope = linreg.slope
        Intercept = linreg.intercept
        r2 = np.square(linreg.rvalue)
        
        while r2>0.99:
            
            # fig,ax = plt.subplots(dpi=200)
            # fig.suptitle('R2 = ' + str(np.round(r2*1000)/1000))
            # ax.plot(intTime,GR_S,'-*',lw=1,ms=2)
            # ax.plot(intTime_linfit,GR_S_linfit,'o',ms=2)
            # ax.plot(intTime_linfit,Intercept + intTime_linfit*Slope,lw=0.5)
            
            # plt.show()
            
            Len += 1
                                    
            intTime_linfit = intTime[0:Len]
            GR_S_linfit = GR_S[0:Len]
            
            linreg = linregress(intTime_linfit,GR_S_linfit)
            
            Slope = linreg.slope
            # print('slope :' + str(Slope))
            Intercept = linreg.intercept
            r2 = np.square(linreg.rvalue)
        
        
        GRmat[50-Len+1:50-Len+1+len(GR_S),ii] = GR_S-GR_S[Len-1]
        
        
        if DebugPlots:
            
            fig0, [ax01,ax02] = plt.subplots(ncols=2, dpi=300)

            ax01.set_title(s + ' - tdeb = ' + str(round(params1[1]/params1[0]*5)/10) + ' min.\n' +
            'T = ' + str(round(params1[0]/60*10)/10)  + ' ' + u"\u00B1" + str(round(stdevs1[0]/60*10)/10) + ' hours.\nR2 = ' 
                          + str(R2_1))
            ax01.plot(Time,AreaC,'*r',ms=3)
            ax01.plot(Time,fitFuncMixed2(Time,params1[0],params1[1],params1[2]),'--b')
            ax01.set_xlabel('Time (min)')
            ax01.set_ylabel('Area')
            # ax01.set_xscale('log')
            # ax01.set_yscale('log')

            ax02.set_title(s + ' - tdeb = ' + str(round(params4[1]/params4[0]*5)/10) + ' min.\n' +
            'T = ' + str(round(params4[0]/60*10)/10)  + ' ' + u"\u00B1" + str(round(stdevs4[0]/60*10)/10) + ' hours.\nR2 = ' 
                          + str(R2_4))
            ax02.plot(Time,AreaC,'*r',ms=3)
            ax02.plot(Time[fitInterval],AreaC[fitInterval],'*g',ms=3)
            ax02.plot(Time,fitFuncMixed2(Time,params4[0],params4[1],params4[2]),'--b',lw=1)
            ax02.set_xlabel('Time (min)')
            ax02.set_ylabel('Area')
            # ax02.set_xscale('log')
            # ax02.set_yscale('log')

            fig0.tight_layout()
            
            fig,[[ax0,ax1],[ax2,ax3]] = plt.subplots(nrows = 2, ncols = 2, dpi = 300)
            
            ax0.plot(Time,AreaC)
            ax0.set_title('Area evolution')
            
            ax1.plot(Time,inv_A)
            ax1.plot(intTime,inv_A_S)
            ax1.set_title('Area inverse')
            
            ax2.plot(intTime,dAdt)
            ax2.plot(intTime,dAdt_S)
            ax2.set_title('area differential')
            
            ax3.plot(intTime,GR,'-*',lw=1,ms=2)
            ax3.plot(intTime,GR_S,'-*',lw=1,ms=2)
            ax3.plot(intTime[-4:],np.ones(4)*GR_2h,'r-')
            ax3.plot(intTime[np.argmin(np.abs(intTime-params4[1]/(2*params4[0])))],GR_S[np.argmin(np.abs(intTime-params4[1]/(2*params4[0])))],'ro',ms=5)
            ax3.plot(intTime[Len-1],GR_S[Len-1],'go',ms=3)
            ax3.set_title('Growth rate local')
            
            fig.tight_layout()
            
            plt.show()

        

        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdeb_full'] = params1[1]/(2*params1[0]) + Delay
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Tau_full'] = params1[0]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'A0fit_full'] = params1[2]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'fitR2_full'] = R2_1
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdeb_inter1'] = params2[1]/(2*params2[0]) + Delay
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Tau_inter1'] = params2[0]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'A0fit_inter1'] = params2[2]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'fitR2_inter1'] = R2_2
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdeb_inter2'] = params3[1]/(2*params3[0]) + Delay
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Tau_inter2'] = params3[0]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'A0fit_inter2'] = params3[2]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'fitR2_inter2'] = R2_3
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdeb_fit'] = params4[1]/(2*params4[0]) + Delay
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdebShift_fit'] = np.argmin(np.abs(Time-params4[1]/(2*params4[0]))) # img shift for alignement on tdeb        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Tau'] = params4[0]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'A0fit'] = params4[2]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'STDtdeb'] = stdevs4[1]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'STDTau'] = stdevs4[0]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'fitR2'] = R2_4
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'ChipRow'] = row
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'GrowthRate'] = GR_2h       
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdeb_GR'] = Time[Len-1] + Delay
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdebShift_GR'] = Len-1 # img shift for alignement on tdeb
        
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdeb'] = params4[1]/(2*params4[0]) + Delay
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdebShift'] = np.argmin(np.abs(Time-params4[1]/(2*params4[0]))) # img shift for alignement on tdeb
        
        
        print('R2 = ' + str(round(R2_4*1000)/1000) + ' - tdeb lin = ' + str(intTime[Len-1]) + ' - tdeb fit = ' + str(params4[1]/params4[0]/2))

    fulltime = np.linspace(0,100,200)-25
    GR_mean = np.nanmean(GRmat,axis = 1)
    
    fig00,ax = plt.subplots(dpi=200)
    fig00.suptitle('Growth rates aligned')
    ax.plot(fulltime,GRmat,lw = 1)
    ax.plot(fulltime,GR_mean,'w--',lw = 2)
    
    
    ax.plot(ax.get_xlim(),[0,0],'r-',lw=1.5)
    
    return(GD)

#%% Osmotic chocs fit

# Fit the evolution of area in time during osmotic chocs. Compression is 
# fitted using a decreasing exponential. Relaxation is fitted using an 
#exponential and a affine component.

# StackList : List of stack names to analyse, CD : dataframe containing contours 
# GD : dataframe containing global (area) data, FPH : Frames per hour,
# ImgStart/End : Images corresponding to times between which to fit compression/relaxation

# Kwargs : 'debug' (True/False) for generating debug plots

def fitOsmoChoc(StackList,CD,GD,FPH,ImgStartComp,ImgEqComp,ImgStartRel,ImgEqRel, **kwargs):
    
    DebugPlots = False
    
    for key, value in kwargs.items(): 
        if key == 'debug':
            DebugPlots = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    print('Sorting data.')
    CD,GD,StackList = sortChocs(CD,GD,StackList,ImgStartComp,ImgEqComp,DebugPlots)    
    CDrel,GDrel,StackListRel = sortChocs(CD,GD,StackList,ImgStartRel,ImgEqRel,DebugPlots)        
    
    for s in StackList:

        print('Fitting curve for : ' + s.ljust(5), end='\n')           
        
        Time = GD.loc[s,'Img'].values.astype(float)/FPH*60 # in minutes
        AreaC = savgol_filter(GD.loc[s,'Area'].values, 5, 3)
        
        
        # data for compression fit
        TimeFitComp = GD.loc[s,'Img'].values.astype(float)[ImgStartComp:ImgEqComp]/FPH*60 # in minutes
        TimeOffset = TimeFitComp[0]
        Time = Time - TimeOffset
        TimeFitComp = TimeFitComp - TimeOffset
        AreaCFitComp = GD.loc[s,'Area'].values[ImgStartComp:ImgEqComp]
               
        fig,ax = plt.subplots(dpi=300)
        ax.plot(Time,AreaC,'*y',ms=3,label='FullData')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Area (mm²)')

        DenseTimeComp = np.linspace(TimeFitComp[0],TimeFitComp[-1],100)

        # fit of compression
        params, cov = curve_fit(f=fitFuncOsmChoc, xdata=TimeFitComp, ydata=AreaCFitComp, 
                                p0=[1, AreaCFitComp[0],AreaCFitComp[0],TimeFitComp[3]],
                                bounds = (0, np.inf), method='trf',loss='soft_l1')

        R2 = np.round(vf.computeR2(AreaCFitComp,fitFuncOsmChoc(TimeFitComp,params[0],params[1],params[2],params[3]))*1000)/1000

        fig.suptitle(s + ' - R2 : ' + str(R2))
        
        # Physical parameters
        DeltaPiOut = 8.314*293*100/1e6 # en MPa, R (gaz parfait) * Temp (K, 20°) * 0.1 (100mM = 100 mol/m3 de choc)
        
        E = params[1]/(params[1]-params[2])*DeltaPiOut # en MPa
        Lh = 1/(params[0]*60*E*1e6) # en m/s/Pa
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'TauFlux'] = params[0] 
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'V0'] = params[1] 
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Veq'] = params[2]   
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'V0-Veq'] = params[1]-params[2]        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Ecomp'] = E             
        GD.loc[(GD.index == s) & (GD['Img'] == 0), '1/Ecomp'] = 1/E       
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'L/H_Comp'] = Lh 
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'H/L_Comp'] = 1/Lh
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Tdeb'] = params[3]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdebShift'] = np.argmin(np.abs(Time-params[3])) # img shift for alignemen
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'fitR2'] = R2
        
        ax.plot(TimeFitComp,AreaCFitComp,'*c',ms=2,label='FittedData')
        ax.plot(DenseTimeComp,fitFuncOsmChoc(DenseTimeComp,params[0],params[1],params[2],params[3]),'--b',lw=1,label='SoftL1')
        
        # Relaxation fit if valid data
        if np.isin(s,StackListRel):
            TimeFitRel = GD.loc[s,'Img'].values.astype(float)[ImgStartRel:ImgEqRel]/FPH*60 # in minutes
            TimeFitRel = TimeFitRel - TimeOffset
            AreaCFitRel = GD.loc[s,'Area'].values[ImgStartRel:ImgEqRel]

            DenseTimeRel = np.linspace(TimeFitRel[0],TimeFitRel[-1],100)

            ax.plot(TimeFitRel,AreaCFitRel,'*m',ms=2,label='FittedDataRel')
        
            paramsRel, covRel = curve_fit(f=fitFuncOsmChoc2, xdata=TimeFitRel, ydata=AreaCFitRel,
                                          p0=[params[0] , params[2],params[1],TimeFitRel[3],0.0005],
                                          bounds = (0, np.inf), method='trf',loss='soft_l1')

            
            R2rel = np.round(vf.computeR2(AreaCFitRel,
                    fitFuncOsmChoc2(TimeFitRel,paramsRel[0],paramsRel[1],paramsRel[2],paramsRel[3],paramsRel[4]))*1000)/1000
            ax.plot(DenseTimeRel,fitFuncOsmChoc2(DenseTimeRel,paramsRel[0],paramsRel[1],paramsRel[2],paramsRel[3],paramsRel[4]),'--r',lw=1,label='SoftL1')
        
        
            fig.suptitle(s + ' - R2Comp : ' + str(R2) + ' - R2Rel : ' + str(R2rel))

            Erel = paramsRel[2]/(paramsRel[2]-paramsRel[1])*DeltaPiOut # en MPa
            LhRel = 1/(paramsRel[0]*60*Erel*1e6) # en m/s/Pa

            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'TauFluxRel'] = paramsRel[0] 
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'V0Rel'] = paramsRel[2] 
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'VeqRel'] = paramsRel[1]   
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'V0Rel-VeqRel'] = paramsRel[2]-paramsRel[1]        
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Erel'] = Erel            
            GD.loc[(GD.index == s) & (GD['Img'] == 0), '1/Erel'] = 1/Erel     
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'L/H_Rel'] = LhRel 
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'H/L_Rel'] = 1/LhRel
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'TdebRel'] = paramsRel[3]
                 
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'E'] = (Erel+E)/2          
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'L/H'] = (Lh+LhRel)/2      
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'H/L'] = 2/(Lh+LhRel) 

            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'fitR2rel'] = R2rel

        fig.tight_layout()
        
        if DebugPlots:
            plt.show()
        else:
            plt.close(fig)
        
        
    
    return(GD)

# Additionnal function : sorting osmotic chocs

# Uses the aspect ratio of gemmae contours in the direction of flux to detect 
# movement du to the rapid change of medium.

def sortChocs(CD,GD,StackList,ImgStart,ImgEq,Plots):

    nppg = len(StackList)
    GoodStacks = []
    AR = []


    for s in StackList:  

        AR.append(
            np.mean(
                np.abs(
                    np.divide(
                        np.divide(GD.loc[s,'Ylength'].values[ImgStart+1:ImgEq-1],
                                  GD.loc[s,'Xlength'].values[ImgStart+1:ImgEq-1]),
                        np.divide(GD.loc[s,'Ylength'].values[ImgStart],
                                  GD.loc[s,'Xlength'].values[ImgStart]))-1
                )
            )
        )

    GoodStacks = list(compress(StackList,AR<2*np.quantile(AR,[0.75])))

    
    if Plots:
        f1,ax1 = plt.subplots(dpi=250,figsize=(5,3.5))
        f2,ax2 = plt.subplots(dpi=250,figsize=(5,3.5)) 

        for s in StackList:
            if s in GoodStacks:
                ax1.plot(GD.loc[s,'Img'].values[ImgStart:ImgEq]/2,GD.loc[s,'Area'].values[ImgStart:ImgEq]/GD.loc[s,'Area'].values[ImgStart],
                         'g-',lw=0.3,zorder=0)
                ax2.plot(GD.loc[s,'Img'].values[ImgStart:ImgEq]/2,np.abs(np.divide(np.divide(GD.loc[s,'Ylength'].values[ImgStart:ImgEq],GD.loc[s,'Xlength'].values[ImgStart:ImgEq]),
                        np.divide(GD.loc[s,'Ylength'].values[ImgStart],GD.loc[s,'Xlength'].values[ImgStart]))-1),'g-',zorder=0,lw=0.3)
            else:            
                ax1.plot(GD.loc[s,'Img'].values[ImgStart:ImgEq]/2,GD.loc[s,'Area'].values[ImgStart:ImgEq]/GD.loc[s,'Area'].values[ImgStart],'r-',
                         lw=0.3,zorder=1)
                ax2.plot(GD.loc[s,'Img'].values[ImgStart:ImgEq]/2, np.abs(np.divide(np.divide(GD.loc[s,'Ylength'].values[ImgStart:ImgEq],GD.loc[s,'Xlength'].values[ImgStart:ImgEq]),
                        np.divide(GD.loc[s,'Ylength'].values[ImgStart],GD.loc[s,'Xlength'].values[ImgStart]))-1),'r-',lw=0.3,zorder=1)


        plt.show()
    print('Kept ' + str(len(GoodStacks)) + '/' + str(nppg) + 'PPGs (' + str(np.round(len(GoodStacks)/nppg*10000)/100) + '%)' )

    return(CD.loc[GoodStacks],GD.loc[GoodStacks],GoodStacks)


#%% Fits validation and sorting

# 1. Comparing iterative fits for growth curve

# GD : dataframe containing global (area) data, label : experiment name

def compareFit(GD, label):
    
    ValuesPos = ~np.isnan(GD['tdeb'].values)
    
    Tdebs_inter1 = GD['tdeb_inter1'].values[ValuesPos]    
    Tdebs_inter2 = GD['tdeb_inter2'].values[ValuesPos]
    Tdebs = GD['tdeb_fit'].values[ValuesPos]
    Tdebs_full = GD['tdeb_full'].values[ValuesPos]
    
    Taus_inter1 = GD['Tau_inter1'].values[ValuesPos]
    Taus_inter2 = GD['Tau_inter2'].values[ValuesPos]
    Taus = GD['Tau'].values[ValuesPos]
    Taus_full = GD['Tau_full'].values[ValuesPos]
    
    A0fits_inter1 = GD['A0fit_inter1'].values[ValuesPos]
    A0fits_inter2 = GD['A0fit_inter2'].values[ValuesPos]
    A0fits = GD['A0fit'].values[ValuesPos]
    A0fits_full = GD['A0fit_full'].values[ValuesPos]
    
    
    MIN = np.min([Tdebs_full, Tdebs, Tdebs_inter1, Tdebs_inter2])
    MAX = np.max([Tdebs_full, Tdebs, Tdebs_inter1, Tdebs_inter2])
    
    fig11, ax11 = plt.subplots(dpi=200)  
    ax11.set_title(label + ' - Tstart (min)')
    ax11.plot([MIN,MAX],[MIN,MAX],'-g')
    ax11.plot(Tdebs_full,Tdebs_inter1,'ro',ms = 5)
    ax11.plot(Tdebs_inter1,Tdebs_inter2,'mo',ms = 4.5)
    ax11.plot(Tdebs_inter2,Tdebs,'o',color=np.divide([150, 131, 236],255),ms = 4)
    ax11.set_xlabel('Previous fit')
    ax11.set_ylabel('Next fit')
    ax11.set_xlim(MIN-0.05*MAX,MAX*1.05)
    ax11.set_ylim(MIN-0.05*MAX,MAX*1.05)
    
    
    MIN = np.min([Taus_full, Taus, Taus_inter1, Taus_inter2])
    MAX = np.max([Taus_full, Taus, Taus_inter1, Taus_inter2])

    fig21, ax21 = plt.subplots(dpi=200)  
    ax21.set_title(label + ' - Tau growth (hours)')
    ax21.plot([MIN,MAX],[MIN,MAX],'-g')
    ax21.plot(Taus_full,Taus_inter1,'ro',ms = 5)
    ax21.plot(Taus_inter1,Taus_inter2,'mo',ms = 4.5)
    ax21.plot(Taus_inter2,Taus,'o',color=np.divide([150, 131, 236],255),ms = 4)
    ax21.set_xlabel('Previous fit')
    ax21.set_ylabel('Next fit')
    ax21.set_xlim(MIN-0.05*MAX,MAX*1.05)
    ax21.set_ylim(MIN-0.05*MAX,MAX*1.05)
    
    
    MIN = np.min([A0fits_full, A0fits, A0fits_inter1, A0fits_inter2])
    MAX = np.max([A0fits_full, A0fits, A0fits_inter1, A0fits_inter2])

    fig31, ax31 = plt.subplots(dpi=200)  
    ax31.set_title(label + ' - A0fit (mm²)')
    ax31.plot([MIN,MAX],[MIN,MAX],'-g')
    ax31.plot(A0fits_full,A0fits_inter1,'ro',ms = 5)
    ax31.plot(A0fits_inter1,A0fits_inter2,'mo',ms = 4.5)
    ax31.plot(A0fits_inter2,A0fits,'o',color=np.divide([150, 131, 236],255),ms = 4)
    ax31.set_xlabel('Previous fit')
    ax31.set_ylabel('Next fit')
    ax31.set_xlim(MIN-0.05*MAX,MAX*1.05)
    ax31.set_ylim(MIN-0.05*MAX,MAX*1.05)
    
  
    
    return

# 2. R2 based selection of data for both growth curve and osmotic chocs fitting

# GD : dataframe containing global (area) data, CD : dataframe containing contours ,
# Th : R2 threshold, label : experiment name

# Kwargs : showHist (True/False) display histogram of validated data in function of R2
# Key : The column name to get the R2 for different fits

def selectR2s(GD, CD, Th, label, **kwargs):
    
    showHist = False
    Key = 'fitR2'
    
    print('Analyzing : ' + label)
    
    for key, value in kwargs.items(): 
        if key == 'showHist':
            showHist = value
        elif key == 'key':
            Key = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    R2sPos = ~np.isnan(GD[Key].values)
    R2s = GD[Key].values[R2sPos]
    goodR2s = R2s>=Th
    frac = np.round(sum(goodR2s)/len(goodR2s)*1000)/10
    
    StackList = GD.index[R2sPos].array.to_numpy()
    goodList = StackList[goodR2s]
    
    goodGD = GD.loc[goodList]
    goodCD = CD.loc[goodList]
    
    if showHist:
        
        
        fig, ax = plt.subplots(dpi=300)
        ax.set_title(label + '\n' + str(frac) + '% of data validated based on R2>' + str(Th))
        n, bins, patches = ax.hist(R2s, bins = np.arange(np.floor(min(R2s)*10)/10, 1.025, 0.025), color = 'r', rwidth = 0.95)
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        ax.hist(R2s[goodR2s], bins = bins, color = 'g', rwidth = 0.95)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        
        plt.show()
        
#     print('Validated data : ' + goodList + '\n\n' )

    return(goodGD, goodCD, R2s, goodList)
