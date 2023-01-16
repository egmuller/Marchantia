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
from scipy.stats import linregress
from scipy.interpolate import interp1d

import copy

import VallapFunc as vf

import matplotlib.pyplot as plt



#%% Fit functions

# Super class for fit objects

class FitMachin:
    
    def __init__(self,t,y,name):
        self.name = name
        self.time = t
        self.values = y
        self.P = []
        self.FI = t<np.max(t+1)
        
    def set_init_fit(self,params):
        self.P_init = params
        self.P = params
        self.R2_init = self.R2()
        
    def set_params(self,params):
        self.P = params
        
    def set_fitinterval(self,FI):
        self.FI = FI
    
    def tdeb(self):        
        return(self.P[0])
    
    def tau(self):        
        return(self.P[1])
    
    def A0(self):        
        return(self.P[2])
    
    def f(self):
        pass
    
    def R2(self):    
        
        R2 = max([0, np.round(vf.computeR2(self.values[self.FI],self.fC()[self.FI])*1000)/1000])
        
        return(R2)
    
    def fC(self):
        pass

# 1. Exponetial growth with a delay + tdeb and R2 extraction functions


class ExpDel(FitMachin):   
    

    def f(self,t,tdeb,T,A0): # Exponential growth with a delay
        
        f = np.multiply(A0,np.exp(np.divide((t-tdeb),T)))
            
        f[t<tdeb] = A0
            
        return(f) 
    
    def fC(self):
        
        return(self.f(self.time,self.P[0],self.P[1],self.P[2]))
      

    
    

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


# A. Function to do iterative fit on a specific window after tdeb, iterations until tdeb has converged

def iterFit(FitClass,name,fitwindow,t,y,params0,Th,maxIter,debug,ax):
    
    FitObj = FitClass(t,y,name)
    
    if debug:
        ax.set_title('Fit : ' + name + ' - FitWindow : ' + str(fitwindow))
        ax.set_xlabel('Old tdeb')
        ax.set_ylabel('New tdeb')
    
    # initial fit on all data -> first tdeb guess
    
    FitObj.set_init_fit(curve_fit(f=FitObj.f, xdata=t, ydata=y, p0=params0, bounds=(0, np.inf), method='trf', loss='soft_l1')[0])
    
    tdebVar = 1
    tdebVars = [1]
    cnt = 0
    
    while (tdebVar>Th) & (cnt<maxIter):
        
        tdeb_old = FitObj.tdeb()
        
        FitObj.set_fitinterval(t<(FitObj.tdeb()+fitwindow*60))
        
        FitObj.set_params(curve_fit(f=FitObj.f, xdata=t[FitObj.FI], ydata=y[FitObj.FI], p0=FitObj.P,                                  
                                  bounds=(0, np.inf), method='trf', loss='soft_l1')[0])
        
        
        
        tdebVar = np.abs((tdeb_old-FitObj.tdeb())/tdeb_old)
        tdebVars.append(tdebVar)
        
        if debug:
            if tdebVar<Th:
                ax.plot(tdeb_old,FitObj.tdeb(),'og',ms=1)
            elif cnt == maxIter-1:
                ax.plot(tdeb_old,FitObj.tdeb(),'xr',ms=2)
            elif cnt == 0:
                ax.plot(tdeb_old,FitObj.tdeb(),'ob',ms=1)
            else:
                ax.plot(tdeb_old,FitObj.tdeb(),'*w',ms=0.5)
            

        cnt += 1       
    print('\nModel for fit : ' + FitObj.name)
    print('Number of iterations : ' + str(cnt))
    print('Final tdeb variations :' + str(tdebVars[-3:]) )
    
    if cnt == maxIter:
        if not all(np.array(tdebVars[-4:])<2*Th):
            print('Thresohold : ' + str(Th) + 'tdebVars :')
            print(tdebVars)
            FitObj.set_fitinterval([])
     
    return(FitObj)
    
    

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

def fitAreaGrowth(StackList,Rows,GD,FPH,Delay,Th, **kwargs):
    
    DebugPlots = False
    Debug = True
    ValidPlots = False
    FitWindow = 15
    
    for key, value in kwargs.items(): 
        if key == 'debug':
            Debug = value
        elif key == 'debugall':
            DebugPlots = value
            Debug = False
        elif key == 'ValidPlots':
            ValidPlots = value
        elif key == 'fitwindow':
            FitWindow = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')

    
    # Matrix for averaging growth rate
    
    GRmat = np.empty((200,len(StackList)))
    
    GRmat[:] = np.nan
            
    if Debug:
        fig1,ax1 = plt.subplots(dpi=200)
      
    else:
        ax1=0
            
    for ii,s,row in zip(range(len(StackList)),StackList,Rows):
        
        print('\n_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')
        print('\nFitting area curve for : ' + s)           
        
        Time = GD.loc[s,'Img'].values.astype(float)/FPH*60 # in minutes
        AreaC = savgol_filter(GD.loc[s,'Area'].values, 11, 2)
        AreaI = interp1d(Time,AreaC,kind = 'quadratic')

        
        ### Growth rate 1/A * dA/dt computation
    
        GR, GR_S, intTime = vf.GrowthRate(AreaC,Time)
        
        GD.loc[s,'GR_Full'] = np.concatenate((GR_S,[np.nan]))
        
        GR_end = np.mean(GR_S[-4:])
        
        ### Computing growth start regime from growth rate
        
        r2 = 1
        Len = 2
        
        intTime_linfit = intTime[0:Len]
        GR_S_linfit = GR_S[0:Len]
        
        linreg = linregress(intTime_linfit,GR_S_linfit)

        
        r2 = np.square(linreg.rvalue)
        
        while r2>0.99:
            
            Slope = linreg.slope
            
            Len += 1
                                    
            intTime_linfit = intTime[0:Len]
            GR_S_linfit = GR_S[0:Len]
            
            linreg = linregress(intTime_linfit,GR_S_linfit)

            r2 = np.square(linreg.rvalue)
            
        
        
        GRmat[50-Len+1:50-Len+1+len(GR_S),ii] = GR_S-GR_S[Len-1]
        
        ### Iterative fits for a convergence of Tdeb with different fits      
        
        FitRes_flat = iterFit(ExpDel,'ExpDel',FitWindow,Time,AreaC,[30,100, AreaC[0]], 0.05, 10, Debug, ax1)
        
        FitResPlot =copy.deepcopy(FitRes_flat)
        
        
        ### Growth rate 1/A * dA/dt computation for fits
        
        if not np.array(FitRes_flat.FI).size == 0:
            GR_flat,FFF,intTime_flat = vf.GrowthRate(FitRes_flat.fC()[FitRes_flat.FI],FitRes_flat.time[FitRes_flat.FI]) 
        else:
            GR_flat,intTime_flat = [0,0]

        if ValidPlots:
            VPlt = FitResPlot.R2() > Th
        else:
            VPlt = True
        
        if DebugPlots & VPlt:
            
            fig0, [ax01,ax02] = plt.subplots(ncols=2, dpi=300, figsize = (7,5))
            
            fig0.suptitle(FitResPlot.name)

            ax01.set_title(s + ' - tdeb = ' + str(round(FitResPlot.P_init[0]*10)/10) +' min.\n' +
            'T = ' + str(round(FitResPlot.P_init[1]/60*10)/10)  + ' hours.\nR2 = ' 
                          + str(FitResPlot.R2_init))
            ax01.plot(Time,FitResPlot.values,'*r',ms=3)
            ax01.plot(Time,FitResPlot.f(Time,*FitResPlot.P_init),'--b')
            ax01.set_xlabel('Time (min)')
            ax01.set_ylabel('Area (mm²)')
            # ax01.set_xscale('log')
            # ax01.set_yscale('log')
    
            ax02.set_title(s + ' - tdeb = ' + str(round(FitResPlot.tdeb()*10)/10) +  ' min.\n' +
            'T = ' + str(round(FitResPlot.tau()/60*10)/10)  +  ' hours.\nR2 = ' 
                          + str(FitResPlot.R2()))
            ax02.plot(Time,AreaC,'*r',ms=3)
            ax02.plot(Time[FitResPlot.FI],AreaC[FitResPlot.FI],'*g',ms=3)
            ax02.plot(Time,FitResPlot.fC(),'--b',lw=1)
            ax02.set_xlabel('Time (min)')
            ax02.set_ylabel('Area (mm²)')
            # ax02.set_xscale('log')
            # ax02.set_yscale('log')
    
            fig0.tight_layout()
            
            fig,[ax0,ax1] = plt.subplots(nrows = 2, dpi = 300)
            
            ax0.plot(intTime,GR*60*24,'-*b',lw=1,ms=2)
            # ax0.plot(intTime,GR_S*60*24,'-c',lw=2)
            # ax0.plot(intTime_linfit[0:-2],Intercept + intTime_linfit[0:-2]*Slope,'--r',lw=2)
            ax0.set_title('Growth rate in time')
            ax0.set_xlabel('Time (min)')
            ax0.set_ylabel('Growth rate (day-1)')
            
            ax1.plot(intTime,GR_S,'-c',lw=1,ms=2)
            ax1.plot(intTime_flat,GR_flat,'--ro',lw=1,ms=2)
            ax1.set_title('Smoothed GR + ExpDel fit')
           
            fig.tight_layout()
            
            plt.show()
            
            print('\nType of fit displayed : ' + FitResPlot.name)
            print('R2 = ' + str(round(FitResPlot.R2()*1000)/1000) + ' - tdeb lin = ' + str(intTime[Len-1]) + ' - tdeb fit = ' + str(FitResPlot.tdeb()))
            

        
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdeb_flat'] = FitRes_flat.tdeb() + Delay
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdebShift_flat'] = np.argmin(np.abs(Time-FitRes_flat.tdeb())) # img shift for alignement on tdeb
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'GrowthAtStart_flat'] = (AreaI(FitRes_flat.tdeb())-AreaC[0])/AreaC[0] # % area increase at tdeb
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'ChipRow'] = row
        # GD.loc[(GD.index == s) & (GD['Img'] == 0), 'H0'] = W0*0.47 -138.3
        
        
        # print('\nH0 estimate : W0 * 0.47 - 138.3 ')
        # print('W0 = ' + str(W0) + ' µm')  
        # print('H0 = ' + str(W0*0.47 -138.3) + ' µm')
        

        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'GR_end'] = GR_end*60*24 # in day-1
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdeb_GR'] = intTime[Len-1] + Delay
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'CaracT_GR'] = 1/np.sqrt(Slope)
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdebShift_GR'] = Len-1 # img shift for alignement on tdeb
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'GrowthAtStart_GR'] = (AreaI(intTime[Len-1])-AreaC[0])/AreaC[0] # % area increase at tdeb
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'fit_name'] = FitResPlot.name
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdeb'] = FitResPlot.tdeb() + Delay
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdebShift'] = np.argmin(np.abs(Time-FitResPlot.tdeb())) # img shift for alignement on tdeb
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Tau'] = FitResPlot.tau()
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'A0fit'] = FitResPlot.A0()
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'fitR2'] = FitResPlot.R2()
        
        
    if Debug:
            
        fulltime = np.linspace(0,100,200)-25
        GR_mean = np.nanmean(GRmat,axis = 1)
        
        fig00,ax = plt.subplots(dpi=200)
        fig00.suptitle('Growth rates aligned')
        ax.plot(fulltime,GRmat,lw = 1)
        ax.plot(fulltime,GR_mean,'w--',lw = 2)
        
        
        ax.plot(ax.get_xlim(),[0,0],'r-',lw=1.5)
        
        plt.show()
    
    return(GD)

#%% Osmotic chocs fit

# Fit the evolution of area in time during osmotic chocs. Compression is 
# fitted using a decreasing exponential. Relaxation is fitted using an 
#exponential and a affine component.

# StackList : List of stack names to analyse, CD : dataframe containing contours 
# GD : dataframe containing global (area) data, FPH : Frames per hour,
# ImgStart/End : Images corresponding to times between which to fit compression/relaxation

# Kwargs : 'debug' (True/False) for generating debug plots

def fitOsmoChoc(StackList,Rows,CD,GD,FPH,ImgStartComp,ImgEqComp,TstartComp,ImgStartRel,ImgEqRel,TstartRel, **kwargs):
    
    DebugPlots = False
    IgnoredCompPoints = []
    IgnoredRelPoints = []
    Concentration = 100 # mM
    
    for key, value in kwargs.items(): 
        if key == 'debug':
            DebugPlots = value
        elif key == 'ignoredCTP':
            IgnoredCompPoints = value
        elif key == 'ignoredRTP':
            IgnoredRelPoints = value
        elif key == 'C_osmo':
            Concentration = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    print('Sorting data.')
    CD,GD,StackList = sortChocs(CD,GD,StackList,ImgStartComp,ImgEqComp,DebugPlots)    
    CDrel,GDrel,StackListRel = sortChocs(CD,GD,StackList,ImgStartRel,ImgEqRel,DebugPlots)        
    
    for s,row in zip(StackList,Rows):

        print('Fitting curve for : ' + s.ljust(5), end='\n')           
        
        Time = GD.loc[s,'Img'].values.astype(float)/FPH*60 # in minutes
        AreaC = GD.loc[s,'Area'].values        
        
        # data for compression fit
        TimeFitComp = GD.loc[s,'Img'].values.astype(float)[ImgStartComp:ImgEqComp]/FPH*60 # in minutes
        TimeOffset = TimeFitComp[0]
        Time = Time - TimeOffset
        TimeFitComp = TimeFitComp - TimeOffset
        AreaCFitComp = AreaC[ImgStartComp:ImgEqComp]
        
        
        TimeFitComp = np.delete(TimeFitComp,IgnoredCompPoints)
        AreaCFitComp = np.delete(AreaCFitComp,IgnoredCompPoints)
               
        fig,ax = plt.subplots(dpi=300)
        ax.plot(Time,AreaC,'*y',ms=3,label='FullData')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Area (mm²)')

        DenseTimeComp = np.linspace(TimeFitComp[0],TimeFitComp[-1],100)

        # fit of compression
        params, cov = curve_fit(f=fitFuncOsmChoc, xdata=TimeFitComp, ydata=AreaCFitComp, 
                                p0=[1, AreaCFitComp[0:TstartComp].mean(),AreaCFitComp[0:TstartComp].mean()*0.98,TimeFitComp[TstartComp]],
                                bounds = (0, np.inf), method='trf',loss='soft_l1')
        
        
        params2, cov2 = curve_fit(f=fitFuncOsmChoc2, xdata=TimeFitComp, ydata=AreaCFitComp, 
                                p0=[1, AreaCFitComp[0:TstartComp].mean(),AreaCFitComp[0:TstartComp].mean()*0.98,TimeFitComp[TstartComp],0],
                                bounds = (0, np.inf), method='trf',loss='soft_l1')

        R2 = np.round(vf.computeR2(AreaCFitComp,fitFuncOsmChoc2(TimeFitComp,params2[0],params2[1],params2[2],params2[3],params2[4]))*1000)/1000

        fig.suptitle(s + ' - R2 : ' + str(R2))
        
        # Physical parameters
        DeltaPiOut = 8.314*298*Concentration/1e6 # en MPa, R (gaz parfait) * Temp (K, 25°) * 0.1 (100mM = 100 mol/m3 de choc)
        
        E = params[1]/np.abs(params[1]-params[2])*DeltaPiOut # en MPa
        Eg = params2[1]/np.abs(params2[2]-params2[1])*DeltaPiOut # en MPa
        LovH = 1/(params2[0]*60*Eg*1e6) # en /s/Pa
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'TauFlux'] = params2[0] 
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'A0'] = params2[1] 
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Aeq'] = params2[2]   
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Ecomp'] = Eg                   
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'EcompNG'] = E  
                        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'L/H0'] = LovH 
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Tdeb'] = params[3]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdebShift'] = np.argmin(np.abs(Time-params[3])) # img shift for alignemen
        
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'GR_InOC'] = params2[4]/params2[1]*60*24 # in day-1
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'fitR2'] = R2
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'fit_name'] = 'Osmotic choc fit'
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'ChipRow'] = row
        
        ax.plot(TimeFitComp,AreaCFitComp,'*c',ms=2,label='FittedData')
        ax.plot(DenseTimeComp,fitFuncOsmChoc2(DenseTimeComp,params2[0],params2[1],params2[2],params2[3],params2[4]),'--g',lw=1,label='SoftL1')
        ax.plot(DenseTimeComp,fitFuncOsmChoc(DenseTimeComp,params[0],params[1],params[2],params[3]),'--b',lw=1,label='SoftL1')
        
        # Relaxation fit if valid data
        if np.isin(s,StackListRel):
            TimeFitRel = GD.loc[s,'Img'].values.astype(float)[ImgStartRel:ImgEqRel]/FPH*60 # in minutes
            TimeFitRel = TimeFitRel - TimeOffset
            AreaCFitRel = AreaC[ImgStartRel:ImgEqRel]            
            
            TimeFitRel = np.delete(TimeFitRel,IgnoredRelPoints)
            AreaCFitRel = np.delete(AreaCFitRel,IgnoredRelPoints)

            DenseTimeRel = np.linspace(TimeFitRel[0],TimeFitRel[-1],100)

            ax.plot(TimeFitRel,AreaCFitRel,'*m',ms=2,label='FittedDataRel')
        
            paramsRel, covRel = curve_fit(f=fitFuncOsmChoc2, xdata=TimeFitRel, ydata=AreaCFitRel,
                                          p0=[params[0] , params[2],params[1],TimeFitRel[TstartRel],0.0005],
                                          bounds = (0, np.inf), method='trf',loss='soft_l1')

            
            R2rel = np.round(vf.computeR2(AreaCFitRel,
                    fitFuncOsmChoc2(TimeFitRel,paramsRel[0],paramsRel[1],paramsRel[2],paramsRel[3],paramsRel[4]))*1000)/1000
            ax.plot(DenseTimeRel,fitFuncOsmChoc2(DenseTimeRel,paramsRel[0],paramsRel[1],paramsRel[2],paramsRel[3],paramsRel[4]),'--r',lw=1,label='SoftL1')
        
        
            fig.suptitle(s + ' - R2Comp : ' + str(R2) + ' - R2Rel : ' + str(R2rel))

            Erel = paramsRel[1]/np.abs(paramsRel[2]-paramsRel[1])*DeltaPiOut # en MPa
            Phi = 1/Eg*(1/paramsRel[0] - 1/params2[0]) # en /MPa/min
            
        
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'TauFluxRel'] = paramsRel[0] 
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'A0Rel'] = paramsRel[1] 
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'AeqRel'] = paramsRel[2]   
            
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Erel'] = Erel  
            
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Phi'] = Phi  
            
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'TdebRel'] = paramsRel[3]
            
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'GrowthSlope'] = paramsRel[4]
            
            GD.loc[(GD.index == s) & (GD['Img'] == 0), 'GR_AfterOC'] = paramsRel[4]/paramsRel[1]*60*24 # in day-1

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

# 1.  R2 based selection of data for both growth curve and osmotic chocs fitting

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
    
    name = GD['fit_name'][0]
    
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
        ax.set_title(label + ' - ' + name + '\n'  + str(sum(goodR2s)) + '/' + str(len(goodR2s)) + '(' + str(frac) + '%) of data validated based on R2>' + str(Th))
        n, bins, patches = ax.hist(R2s, bins = np.arange(np.floor(min(R2s)*10)/10, 1.025, 0.025), color = 'r', rwidth = 0.95)
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        ax.hist(R2s[goodR2s], bins = bins, color = 'g', rwidth = 0.95)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        
        plt.show()
        
#     print('Validated data : ' + goodList + '\n\n' )

    return(goodGD, goodCD, R2s, goodList)
