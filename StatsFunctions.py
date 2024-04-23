# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:43:06 2022

@author: laplaud
"""

# Imports
from scipy.stats import ranksums, linregress, kruskal

import numpy as np
import numpy.matlib as mtl
import pandas as pd

import statsmodels.api as sm
from statsmodels.formula.api import ols

import matplotlib.pyplot as plt
import seaborn as sns

#%% Compute and plot ranksum significativity between two distributions

# ax : axes to plots on, hmax : level to plot the first bar at, step : increment 
# of distance for fullstep, fullstep : offset from hmax to plot at that is incremented
# data1/2 : datasets to compare, pos1/2 : x-position of the data representation

def plotSig(ax,hmax,step,fullstep,data1,data2,pos1,pos2):
    
    s,p = ranksums(data1,data2)
    mad = np.mean(np.abs(data1-np.median(data1)))
    ES = (np.median(data2)-np.median(data1))/mad
    RC = (np.median(data2)-np.median(data1))/np.median(data1)*100
                        
    if pos2 == pos1+1: 
        h = np.median([np.median(data1),np.median(data2)])
        ax.plot([pos1+0.2, pos2-0.2], [h ,h], 'k-',zorder=0)
        ax.text((pos1+pos2)/2,h+1*step,'p = ' + str(np.round(p*1000)/1000) + '\nRC = '+str(round(RC))+ '\%\nES = ' + str(np.round(ES*10)/10), ha='center',fontsize='x-small')
        ax.set_ylim(top=hmax+fullstep+step)
            
    else:       
        h = hmax
        fullstep += step
        ax.plot([pos1, pos2], [h+fullstep ,h+fullstep], 'k-',zorder=0)
        #ax.text((pos1+pos2)/2,h+fullstep+0.2*step,'p = ' + str(round(p*1000)/1000), ha='center',fontsize='small')
        ax.text((pos1+pos2)/2,h+fullstep+0.2*step,'p = ' + str(np.round(p*1000)/1000)  + '\nRC = '+str(round(RC))+ '\% - ES = ' + str(np.round(ES*10)/10), ha='center',fontsize='x-small')
        ax.set_ylim(top=h+fullstep+step)

    return(fullstep)


#%% Computes and displays correlation between input variables

# GDs : dataframe containing variables to correlate, 
# labels : conditions name for each dataframe


# Kwargs : 'corrmethod' ('pearson'/'kendall'/'spearman') choice of correlation type to use,
# 'dfcols' (string name of dataframe columns) data to correlate, 'columnslabels' (string list) name 
# to use for each columns chosen by dfcols, 'PlotFits' (True/False), 'colors' (RGB 0-1 triplets)

def Corr(GDs,labels, **kwargs):
    
    corrmethod = 'pearson'
    dfcols = ['A0fit','Area','Tau','tdeb']
    colslab = dfcols
    colors = mtl.repmat([0.8, 0, 0.7],len(GDs),1)
    PlotFits = False

    
    for key, value in kwargs.items(): 
        if key == 'corrmethod':
            corrmethod = value 
        elif key == 'columns':
            dfcols = value
            colslab = dfcols
        elif key == 'columnslabels':
            colslab = value
        elif key == 'colors':
            colors = value
        elif key == 'PlotFits':
            PlotFits = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')

        
    for GD,lab,colo in zip(GDs,labels,colors) :
        
        if lab[0:6] == 'Pooled':
            GDtoCorr = GD.loc[GD['Img'] == 0, [*dfcols,'Expe']]
        else:
            GDtoCorr = GD.loc[GD['Img'] == 0, dfcols]
        corrMat = GDtoCorr.corr(method=corrmethod)
        
        plt.figure(dpi=250,facecolor = 'white')
        plt.title(corrmethod + ' correlation for \n' + lab)
        mask = np.zeros_like(corrMat)
        mask[np.tril_indices_from(mask,k=-1)] = True
        sns.heatmap(corrMat,mask = mask,square=True,vmin=-1,vmax=1,annot=True,fmt=".3f",annot_kws={"size":8}) #,cmap = 'YlGnBu'
        
        if PlotFits:
            
            if corrmethod=='pearson':
                plotkind = 'reg'
            else:
                plotkind = 'scatter'
        
            for i in range(0,len(dfcols)-1):
                for j in range(i+1,len(dfcols)):
                    
                    x,y = (GDtoCorr[dfcols[i]],GDtoCorr[dfcols[j]])

                    mask = ~np.isnan(x) & ~np.isnan(y)
                    
                    linreg = linregress(x[mask],y[mask])
                    

                    if lab[0:6] == 'Pooled':
                        
                        g = sns.jointplot(data=GDtoCorr,x=dfcols[i],y=dfcols[j],kind='scatter',hue = 'Expe',height = 12, palette = colors[1:])
                        
                        g.ax_joint.legend(fontsize='xx-large')
                        
                    else:
    
                        g = sns.jointplot(x=x[mask],y=y[mask],kind=plotkind,color = colo,height = 12)
                        
                        if corrmethod=='pearson':
                            g.ax_joint.legend([f"S = {linreg.slope:.2f}",
                                           f"CC = {linreg.rvalue:.3f}\nP = {linreg.pvalue:.3f}"],
                                          fontsize='xx-large')
                    
                    g.fig.suptitle('Correlation between ' + dfcols[i] + ' and ' + dfcols[j] +
                               '.\n Experiment : ' + lab + ' - n = ' + str(len(x[mask])),fontsize=30)
                    g.fig.patch.set_facecolor('white')
                    g.ax_joint.set_xlabel(colslab[i],fontsize = 35)
                    g.ax_joint.set_ylabel(colslab[j],fontsize = 35)
                    g.ax_joint.tick_params(axis='both', labelsize=30)
                    
                    g.fig.tight_layout() 
    

     
#%% Compute Two-way ANOVA

# var : Variable on which to do analysis (in GDs), catgs1/2 : categories identifiers
# GDs dataframe to compare data from

def TwowayANOVA(var,catgs1,catgs2,GDs):
    
    FullANOVAdataset = pd.DataFrame(data=None,columns=['Date','Condition',var]) 
    
    # create complete data set
    for C1,C2,GD in zip(catgs1,catgs2,GDs):
        
        values = GD.loc[(GD['Img'] == 0),var].values
        nsample = np.size(values)

        ANOVAdataset = pd.DataFrame({'Date':np.repeat(C1,nsample),'Condition':np.repeat(C2,nsample),var:values})
        
        FullANOVAdataset = FullANOVAdataset.append(ANOVAdataset)
    
    #perform two-way ANOVA
    model = ols(var + ' ~ C(Date) + C(Condition) + C(Date):C(Condition)', data=FullANOVAdataset).fit()
    res = sm.stats.anova_lm(model, typ=2)
    
    print('Two way ANOVA for : ' + var)
    print(res)
    print('\n')

    return(res)
    

#%% Kruskal-wallis test on 2+ samples

def StatsKruskal(ax,data):

    h,p = kruskal(*data, nan_policy = 'omit')
    
    ax.set_title('Kruskal-wallis test : p = ' + str(np.round(p*1000)/1000))