# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:22:26 2021

@author: Valentin Laplaud

Useful generic functions 

"""

hfkgkgkghgk
import numpy as np
import pandas as pd
import mpmath as mpm

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import directed_hausdorff 
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

import cv2 as cv

#  1. Compute normal to a vector

# Normal vector is normalized, and by default rotated counter clockwise from given vector
# the input vector is from (x1,y1) to (x2,y2)


def getNormal(x1,y1,x2,y2, **kwargs):
    
    rotation = 'CCW'
    
    for key, value in kwargs.items(): 
        if key == 'rotation':
            rotation = value
        else:            
            print('Unknown key : ' + key+ '. Kwarg ignored.')
            
    dx = x2 - x1
    dy = y2 - y1
    
    Norm = np.sqrt(np.square(dx) + np.square(dy))
    
    dxN = np.divide(dx,Norm)
    dyN = np.divide(dy,Norm)
    
    if rotation == 'CCW':
        
        x = -dyN
        y = dxN
        
    elif rotation == 'CW':
        
        x = dyN
        y = -dxN

    else:
        print('Wrong rotation parameter !! Should be ''CW'' or ''CCW''. Default is ''CCW'' with no input.')
            
    return(x,y) 
    
#  2. Plotting boxplots with data points on top 

# A function combining boxplot with seaborn's swarmplot for a better display of data

def boxswarmplot(Title,Ylabel,Data,facecolors,Labels,**kwargs):

    fig,ax = plt.subplots(dpi = 250,facecolor='black',figsize = (5,3.5))
    fig.suptitle(Title)
 
    
    cap= [None]*len(Data)
    med= [None]*len(Data)
    
    grouping = []
    
    for dat,col,lab,i in zip(Data,facecolors,Labels,range(len(Data))):
    
        # plots properties
        plotprops = {'color':'white'}
        boxprops = {'color':'white','facecolor':col}
        
        lab = lab + '\nn = ' + str(len(dat))
        
        Labels[i] = lab

        bp = ax.boxplot(dat, positions = [i], labels = [lab],patch_artist =True, boxprops=boxprops, capprops =plotprops,
                    showfliers=False,whiskerprops=plotprops,medianprops =plotprops)
        
        grouping = np.append(grouping,np.ones(len(dat))*i)
    
        cap[i] = bp['caps'][1].get_ydata(orig=True)[0]
        med[i] = bp['medians'][0].get_ydata(orig=True)[0]
    
    sns.swarmplot(x=grouping,y=pd.concat(Data),color = 'white', size=2, ax = ax)
    
    ax.set_ylabel(Ylabel)
    
    ax.set_xticklabels(Labels)
    
    return(fig,ax,cap,med)
    

# 3. Coordinate conversion from cartesian to circular (in deg) an vice versa

def ToCirc(X,Y, **kwargs):
    
    Angle = 'rad'
    
    for key, value in kwargs.items(): 
        if key == 'angle':
            Angle = value
        else:            
            print('Unknown key : ' + key+ '. Kwarg ignored.')
    
    
    if Angle == 'deg':
        Alpha = np.rad2deg(np.arctan2(Y,X))
    elif Angle == 'rad':
        Alpha = np.arctan2(Y,X)
    else:
        print('Wrong angle unit : ' + Angle + '. Default to radians.')         
        Alpha = np.arctan2(Y,X)
        
    Radius = np.sqrt(np.square(X)+np.square(Y))
    
    return(Alpha,Radius)



def ToCart(Alpha,Radius, **kwargs):
    
    Angle = 'rad'
    
    for key, value in kwargs.items(): 
        if key == 'angle':
            Angle = value
        else:            
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    if Angle == 'deg':
        Alpharad = np.deg2rad(Alpha)
    elif Angle == 'rad':
        Alpharad = Alpha
    else:
        print('Wrong angle unit : ' + Angle + '. Default to radians.') 
        Alpharad = Alpha
    
    
    X = Radius*np.cos(Alpharad)
    Y = Radius*np.sin(Alpharad)
    
    return(X,Y)

# 4.1 Euclidian distance between two arrays of points in carthesian coordinates 
def dist(x1,y1,x2,y2):
    
    d = np.sqrt(np.square(x1-x2)+np.square(y1-y2))
    
    return(d)

# 4.2 Computation of Hausdorff distance (https://en.wikipedia.org/wiki/Hausdorff_distance) between two contours
    
    
def HausdorffDist(x1,y1,x2,y2, **kwargs):
    
    DebugPlots = False

    for key, value in kwargs.items():
        if key == 'debug':
            DebugPlots = value
    
    c1 = [[x,y] for x,y in zip(x1,y1)]
    c2 = [[x,y] for x,y in zip(x2,y2)]
    
    d1, i11, i12 = directed_hausdorff(c1, c2)
    d2, i21, i22 = directed_hausdorff(c2, c1)
    D = max(d1,d2)
    
    
    if DebugPlots:   
        f0, ax0 = plt.subplots(dpi=200,facecolor='white')
        ax0.set_aspect('equal', adjustable='box')
        ax0.plot(x1,y1,'r.')
        ax0.plot(x2,y2,'b.')
        ax0.plot(x1[i11],y1[i11],'*g')
        ax0.plot(x2[i12],y2[i12],'*g')
        ax0.plot([x1[i11],x2[i12]],[y1[i11],y2[i12]],'g')
        ax0.plot(x1[i21],y1[i21],'*c')
        ax0.plot(x2[i22],y2[i22],'*c')
        ax0.plot([x1[i21],x2[i22]],[y1[i21],y2[i22]],'c')
         
   
    return(D)

# 5. Intersection volume between sphere and cylinder

def interVolSC(Rs,Rc,Dsc):
    
                
    """ ref : Boersma and Kamminga, 1961. (https://core.ac.uk/download/pdf/82412251.pdf) """
    """ Equation (5) and (8) are implemented with the use of mpmath for computation of the elliptic integrals. """
    
    # Rs sphere radius, Rc cylinder radius, Dsc distance between the two centers
    
    # normalization to sphere radius
    rho = Rc/Rs
    eta = Dsc/Rs
    
    """ Formula is valid if sphere and cylinder intersects """
    if not (eta-rho) < 1:
        raise ValueError('Invalid case ! This code''s formula is only valid for intersecting sphere and cylinders \n -> (Dsc/Rs) - (Rc/Rs) < 1 !!')
        

    """ Heuman's lambda function """
    def Lambda0(beta,m):      
        """ From (https://link.springer.com/content/pdf/10.1007%2F978-3-642-65138-0.pdf) form 150.3 page 36 """
        L = 2/np.pi*(mpm.ellipe(m)*mpm.ellipf(beta,(1-m))+mpm.ellipk(m)*mpm.ellipe(beta,(1-m))-mpm.ellipk(m)*mpm.ellipf(beta,(1-m)))
        return(L)   
    
    
    if (eta+rho)>1 :
        """ Formula (5) is valid for eta+rho > 1 , this means that the part of the cylinder is outside the drop  """ 
        
        m = (1-(eta-rho)**2)/(4*rho*eta) # parameter for eliptic functions of mpmath (=k² in the paper)    
        theta = np.arcsin(eta-rho)  
        
        V = (2/3*np.pi*(1-Lambda0(theta,m)) 
             -8/9*np.sqrt(rho*eta)*(6*rho**2+2*rho*eta-3)*(1-m)*mpm.ellipk(m) 
             +8/9*np.sqrt(rho*eta)*(7*rho**2+eta**2-4)*mpm.ellipe(m))
    
    else:
        """ Formula (8) is valid for eta+rho <= 1 , this means that the cylinder is completely inside the drop  """ 
        
        m = (4*rho*eta)/(1-(eta-rho)**2) # parameter for eliptic functions of mpmath (=k² in the paper)    
        theta = np.arcsin((eta-rho)/(eta+rho))  
        
        V = (2/3*np.pi*(1-Lambda0(theta,m))
             -(4*np.sqrt(1-(eta-rho)**2))/(9*(eta+rho))*(2*rho-4*eta+(eta+rho)*(eta-rho)**2)*(1-m)*mpm.ellipk(m)
             +4/9*np.sqrt(1-(eta-rho)**2)*(7*rho**2+eta**2-4)*mpm.ellipe(m))

    
    return(float(V*Rs**3))

# 6. simple ismember function, checks if A is within B
def ismember(A, B):
    return [ np.sum(b == A) for b in B ]

# 7. R2 computation for a fit
def computeR2(Ydata,Yfit):
    # Ydata are the fitted data, Yfit the comuted value from the fit
    
    SumResidues = np.sum(np.square(np.subtract(Ydata,Yfit)))
    TotalVariance = np.sum(np.square(np.subtract(Ydata,np.mean(Ydata))))
    
    R2 = 1 - SumResidues/TotalVariance
    
    return R2
    
    
 # 8. Normalisation of a contour by its area

def normaliseContourByArea(X,Y):
    
    Cnt = np.array([[x,y] for x,y in zip(X,Y)], dtype=np.float32)
        
    area = cv.contourArea(Cnt)
    
  #  print('Contour original area : ' + str(area))

    XNorm = np.divide(X,np.sqrt(area))
    YNorm = np.divide(Y,np.sqrt(area))

    # CntNorm = np.array([[x,y] for x,y in zip(XNorm,YNorm)], dtype=np.float32)

    # areaNorm = cv.contourArea(CntNorm)
    
 #   print('Contour normalized area : ' + str(areaNorm))
        
    return(XNorm,YNorm)


# 9. Compute growth rate (1/A * dA/dt) for a curve A and time T 

def GrowthRate(A,Time):
    
    dA = np.diff(A)        
    dt = np.diff(Time)
    
    dAdt = np.divide(dA,dt)
    dAdt_S = savgol_filter(dAdt, 11, 3)    
    
    intTime = Time[0:-1]+dt/2
    
    inv_A = np.divide(1,A)
    inv_f = interp1d(Time,inv_A)
    inv_A_timed = inv_f(intTime)
    
    inv_A_S = savgol_filter(inv_A_timed,11, 3)
    
    
    GR = np.multiply(inv_A_timed,dAdt)
    GR_S = np.multiply(inv_A_S,dAdt_S)
    
    return(GR,GR_S,intTime)
        

        