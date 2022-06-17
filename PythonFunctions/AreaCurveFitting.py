# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:03:52 2022

@author: laplaud
"""


# Imports
import numpy as np

from scipy.optimize import curve_fit

import VallapFunc as vf

import matplotlib.pyplot as plt



#%% Fit functions

# 1. Exponetial growth wirh a delay

def fitFunc(t,T,tdeb,A0): # Exponential growth with a delay
    
    f = np.multiply(A0,np.exp(np.divide((t-tdeb),T)))
        
    f[t<tdeb] = A0
        
    return(f)


# 2. Exponential decay from initial volume to equilibrium volume

def fitFuncOsmChoc(t,T,A0,Aeq,tdeb): # 
    
    f = np.multiply((A0-Aeq),np.exp(np.divide(-(t-tdeb),T))) + Aeq
        
    f[t<tdeb] = A0
    
    return(f)

# 3. Exponential decay from initial volume to equilibrium volume + affine component

def fitFuncOsmChoc2(t,T,A0,Aeq,tdeb,B): 
    
    f = np.multiply((A0-Aeq),np.exp(np.divide(-(t-tdeb),T))) + Aeq + np.multiply(t,B)
        
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
            
    for s,row in zip(StackList,Rows):
        
        print('Fitting area curve for : ' + s.ljust(5), end='\n')           
        
        Time = GD.loc[s,'Img'].values.astype(float)/FPH*60 # in minutes
        AreaC = GD.loc[s,'Area'].values
        
        # first fit to determine tdeb
        params1, cov1 = curve_fit(f=fitFunc, xdata=Time, ydata=AreaC, p0=[100, 30, AreaC[0]], bounds=(0, np.inf), method='trf', loss='soft_l1')
        stdevs1 = np.sqrt(np.diag(cov1))
        
        R2_1 = np.round(vf.computeR2(AreaC,fitFunc(Time,params1[0],params1[1],params1[2]))*1000)/1000
                        
        # Second fit, only until 15 hours after the start of growth
        fitInterval = Time<(params1[1]+FitWindow*60)
        params2, cov2 = curve_fit(f=fitFunc, xdata=Time[fitInterval], ydata=AreaC[fitInterval], p0=params1,
                                  bounds=(0, np.inf), method='trf', loss='soft_l1')
        
        
        stdevs2 = np.sqrt(np.diag(cov2))
        
        R2_2 = np.round(vf.computeR2(AreaC[fitInterval],fitFunc(Time[fitInterval],params2[0],params2[1],params2[2]))*1000)/1000
        
        # Third fit, only until 15 hours after the start of growth, to confirm second
        fitInterval = Time<(params2[1]+FitWindow*60)
        params3, cov3 = curve_fit(f=fitFunc, xdata=Time[fitInterval], ydata=AreaC[fitInterval], p0=params2,
                                  bounds=(0, np.inf), method='trf', loss='soft_l1')
        
        
        stdevs3 = np.sqrt(np.diag(cov2))
        
        R2_3 = np.round(vf.computeR2(AreaC[fitInterval],fitFunc(Time[fitInterval],params3[0],params3[1],params3[2]))*1000)/1000
        
        
        # Fourth fit, only until 15 hours after the start of growth, to confirm third
        fitInterval = Time<(params3[1]+FitWindow*60)
        params4, cov4 = curve_fit(f=fitFunc, xdata=Time[fitInterval], ydata=AreaC[fitInterval], p0=params3,
                                  bounds=(0, np.inf), method='trf', loss='soft_l1')
        
        
        stdevs4 = np.sqrt(np.diag(cov2))
        
        R2_4 = np.round(vf.computeR2(AreaC[fitInterval],fitFunc(Time[fitInterval],params4[0],params4[1],params4[2]))*1000)/1000
        
        
        
        fig, [ax1,ax2] = plt.subplots(ncols=2, dpi=300)

        ax1.set_title(s + ' - tdeb = ' + str(round(params1[1]*10)/10) + ' ' + u"\u00B1" + str(round(stdevs1[1]*10)/10) + ' min.\n' +
        'T = ' + str(round(params1[0]/60*10)/10)  + ' ' + u"\u00B1" + str(round(stdevs1[0]/60*10)/10) + ' hours.\nR2 = ' 
                      + str(R2_1))
        ax1.plot(Time,AreaC,'*r',ms=3)
        ax1.plot(Time,fitFunc(Time,params1[0],params1[1],params1[2]),'--b')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Area')

        ax2.set_title(s + ' - tdeb = ' + str(round(params2[1]*10)/10) + ' ' + u"\u00B1" + str(round(stdevs2[1]*10)/10) + ' min.\n' +
        'T = ' + str(round(params2[0]/60*10)/10)  + ' ' + u"\u00B1" + str(round(stdevs2[0]/60*10)/10) + ' hours.\nR2 = ' 
                      + str(R2_2))
        ax2.plot(Time,AreaC,'*r',ms=3)
        ax2.plot(Time[fitInterval],AreaC[fitInterval],'*g',ms=3)
        ax2.plot(Time,fitFunc(Time,params4[0],params4[1],params4[2]),'--b',lw=1)
        ax2.set_xlabel('Time (min)')
        ax2.set_ylabel('Area')

        fig.tight_layout()
        
        if DebugPlots:
            plt.show()
        else:
            plt.close(fig)
        

        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdeb_full'] = params1[1] + Delay
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdebShift_full'] = np.argmin(np.abs(Time-params1[1])) # img shift for alignement on tdeb
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Tau_full'] = params1[0]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'A0fit_full'] = params1[2]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'STDtdeb_full'] = stdevs1[1]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'STDTau_full'] = stdevs1[0]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'fitR2_full'] = R2_1
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdeb_inter1'] = params2[1] + Delay
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdebShift_inter1'] = np.argmin(np.abs(Time-params2[1])) # img shift for alignement on tdeb
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Tau_inter1'] = params2[0]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'A0fit_inter1'] = params2[2]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'STDtdeb_inter1'] = stdevs2[1]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'STDTau_inter1'] = stdevs2[0]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'fitR2_inter1'] = R2_2
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdeb_inter2'] = params3[1] + Delay
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdebShift_inter2'] = np.argmin(np.abs(Time-params3[1])) # img shift for alignement on tdeb
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Tau_inter2'] = params3[0]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'A0fit_inter2'] = params3[2]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'STDtdeb_inter2'] = stdevs3[1]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'STDTau_interé'] = stdevs3[0]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'fitR2_inter2'] = R2_3
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdeb'] = params4[1] + Delay
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'tdebShift'] = np.argmin(np.abs(Time-params4[1])) # img shift for alignement on tdeb
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Tau'] = params4[0]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'A0fit'] = params4[2]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'STDtdeb'] = stdevs4[1]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'STDTau'] = stdevs4[0]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'fitR2'] = R2_4
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'ChipRow'] = row
        
        
        ## Additionnal growth fitting (end-10h -> end) for correlation with osmotic chocs
        
        def expfitf(t,Tau):
            return(np.exp(np.divide(t,Tau)))
        
        params, cov = curve_fit(f=expfitf, xdata=Time[-20:len(Time)], ydata=AreaC[-20:len(Time)]/params4[2], p0=params4[0],
                                  bounds=(0, np.inf), method='trf', loss='soft_l1')
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Tau10h'] = params[0]
        GD.loc[(GD.index == s) & (GD['Img'] == 0), '1/Tau10h'] = 1/params[0]
        
        params, cov = curve_fit(f=expfitf, xdata=Time[-10:len(Time)], ydata=AreaC[-10:len(Time)]/params4[2], p0=params[0],
                                  bounds=(0, np.inf), method='trf', loss='soft_l1')
        
        GD.loc[(GD.index == s) & (GD['Img'] == 0), 'Tau5h'] = params[0]

    
    return(GD)

#%% Osmotic chocs fit
