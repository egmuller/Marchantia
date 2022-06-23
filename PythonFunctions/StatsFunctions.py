# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:43:06 2022

@author: laplaud
"""

# Imports
from scipy.stats import ranksums

import numpy as np


#%% Compute and plot ranksum significativity between two distributions

# ax : axes to plots on, hmax : level to plot the first bar at, step : increment 
# of distance for fullstep, fullstep : offset from hmax to plot at that is incremented
# data1/2 : datasets to compare, pos1/2 : x-position of the data representation

def plotSig(ax,hmax,step,fullstep,data1,data2,pos1,pos2):
    
    s,p = ranksums(data1,data2)
                        
    if pos2 == pos1+1: 
        h = np.median([np.median(data1),np.median(data2)])
        ax.plot([pos1+0.2, pos2-0.2], [h ,h], 'w-',zorder=0)
        ax.text((pos1+pos2)/2,h+0.2*step,'p = ' + str(round(p*1000)/1000), ha='center',fontsize='small')
        ax.set_ylim(top=hmax+fullstep+step)
            
    else:       
        h = hmax
        fullstep += step
        ax.plot([pos1, pos2], [h+fullstep ,h+fullstep], 'w-',zorder=0)
        ax.text((pos1+pos2)/2,h+fullstep+0.2*step,'p = ' + str(round(p*1000)/1000), ha='center',fontsize='small')
        ax.set_ylim(top=h+fullstep+step)

    return(fullstep)
