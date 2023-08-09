#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:32:04 2023

@author: emuller
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
from skimage import io
import cv2 as cv
from PIL import Image
from IPython import get_ipython

from scipy.spatial import ConvexHull
from skimage.morphology import binary_opening, binary_closing, remove_small_holes

from ContourAnalysis import getLandmarks

import VallapFunc as vf

import tifffile as tifff

from GemmaeDetection import create_circular_mask

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#%% Utility functions

def add_border(Matrix, dim):
        dim_mat = np.shape(Matrix)
        border = cv.copyMakeBorder(Matrix,
                                    top=(dim[0] - dim_mat[0])//2,
                                    bottom=(dim[0] - dim_mat[0])//2 + (dim[0] - dim_mat[0])%2,
                                    left=(dim[1] - dim_mat[1])//2,
                                    right=(dim[1] - dim_mat[1])//2 + (dim[1] - dim_mat[1])%2,
                                    borderType=cv.BORDER_CONSTANT,
                                    value=[0,0,0] ) 
        return border
    
def add_border_left(Matrix, size):
        border = cv.copyMakeBorder(Matrix,
                                    top=0,
                                    bottom=0,
                                    left=size,
                                    right=0,
                                    borderType=cv.BORDER_CONSTANT,
                                    value=[255,255,255] ) 
        return border
    
import math

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

#%% Select appropriate gemmae for optical flow

def sortGemmae(RGBimg, s, size = 300):
        
    get_ipython().run_line_magic('matplotlib', 'qt')
    
    f,ax = plt.subplots(dpi=250)
    ax.set_title("Select option for" + str(s), color = 'black', fontsize=25)
    
    
    
    rect1 = mpl.patches.Rectangle((5, 50), 30, 30, linewidth=2, edgecolor='r', facecolor='none')
    rect2 = mpl.patches.Rectangle((5, 150), 30, 30, linewidth=2, edgecolor='r', facecolor='none')
    rect3 = mpl.patches.Rectangle((5, 250), 30, 30, linewidth=2, edgecolor='r', facecolor='none')
    
    # Add the patch to the Axes
    ax.add_patch(rect1)
    plt.text(45, 70, 'OK', c='r')
    
    ax.add_patch(rect2)
    plt.text(45, 170, 'Confined', c='r')
    
    ax.add_patch(rect3)
    plt.text(45, 270, 'Not 2 notches', c='r')
    
    ax.imshow(RGBimg)
    
    pts = np.asarray(plt.ginput(1, timeout=-1))
    
    plt.close()
    get_ipython().run_line_magic('matplotlib', 'inline')

    if pts[0][1] < 85 and pts[0][1] > 45 and pts[0][0] < 40 and pts[0][0] > 0:
        sort = 'OK'
    elif pts[0][1] < 185 and pts[0][1] > 145 and pts[0][0] < 40 and pts[0][0] > 0:
        sort = 'Confined'
    elif pts[0][1] < 285 and pts[0][1] > 245 and pts[0][0] < 40 and pts[0][0] > 0:
        sort = 'Not2Notches'
    else :
        return False, 0
    
    return True, sort

def selectGemmae(P, StackList, size = 300, nimg = 10):
    
    OpticalStack = []
    Not2notches = []
    Confined = []
    
    for s in StackList :
        RGBimg = io.imread(P + '/' + s + '.tif', key = nimg)
        RGBimg = add_border_left(RGBimg, size = size)
        Done = False
        
        while not Done :
        
            Done, sort = sortGemmae(RGBimg, s, size = size)
        
        if sort == 'OK':
           OpticalStack.append(s) 
        elif sort == 'Confined':
           Confined.append(s) 
        elif sort == 'Not2Notches':
           Not2notches.append(s) 
        

    return OpticalStack, Not2notches, Confined

#%% Get points of interest

def getPointsCoordinates_alignement_Attach(P,s,npts,nimg,Title):            
    #P = path, s = stacks, npts = number of point, nimg = number of image, title = instructions    
    RGBimg = io.imread(P + '/' + s + '.tif', key = nimg) # get the  image from tiff stack

    # ask user to click
    get_ipython().run_line_magic('matplotlib', 'qt')
    f,ax = plt.subplots(dpi=250)
    ax.set_title(Title, color = 'black', fontsize=25)
    ax.imshow(RGBimg)
    pts = np.asarray(plt.ginput(npts, timeout=-1))

    plt.close()
    get_ipython().run_line_magic('matplotlib', 'inline')

    Points = pts[:]

    for i in range(npts):
        x = pts[i][0]
        y = pts[i][1]

        Points[i][0] = x
        Points[i][1] = y

    return(Points)


def getPointsCoordinates_alignementNotch(P,s,npts,nimg,Title):            
    #P = path, s = stacks, npts = number of point, nimg = number of image, title = instructions
    
    RGBimg = io.imread(P + '/' + s + '.tif', key = nimg) # get the  image from tiff stack

    # ask user to click
    get_ipython().run_line_magic('matplotlib', 'qt')
    f,ax = plt.subplots(dpi=250)
    ax.set_title(Title, color = 'black', fontsize=25)
    ax.imshow(RGBimg)
    if nimg == 0 :
        pts = np.asarray(plt.ginput(npts, timeout=-1))
    
    # to show the position of the previous notches
    if nimg > 0 :
        RGBimg_prev = io.imread(P + '/' + s + '.tif', key = nimg-1)
        #RGBimg_prev = cv.cvtColor(RGBimg_prev, cv.COLOR_BGR2GRAY)
        ax.imshow(RGBimg_prev, alpha = 0.3)
        
        RefPts = pd.read_csv(P + '/clickedpoints_alltimes.csv', index_col = 'Ind')
        X1 = RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == nimg-1) ,'Xnotch1']
        X2 = RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == nimg-1) ,'Xnotch2']
        Y1 = RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == nimg-1) ,'Ynotch1']
        Y2 = RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == nimg-1) ,'Ynotch2']
        ax.text(X1, Y1, 'N1')
        ax.text(X2, Y2, 'N2')
        
        pts = np.asarray(plt.ginput(npts, timeout=-1))
    

    plt.close()
    get_ipython().run_line_magic('matplotlib', 'inline')

    Points = pts[:]

    for i in range(npts):
        x = pts[i][0]
        y = pts[i][1]

        Points[i][0] = x
        Points[i][1] = y

    return(Points)



def getNotchesAttach_initial(P,StackList, **kwargs):
    
    # Redo notches detection?
    overwrite = False
    
    for key, value in kwargs.items(): 
        if key == 'overwrite':
            overwrite = value      
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    # Load or create dataframe
    if os.path.exists(P + '/clickedpoints_alltimes.csv'):
        RefPts = pd.read_csv(P + '/clickedpoints_alltimes.csv', index_col = 'Ind')  
    else:                
        RefPts = pd.DataFrame(data=None,columns=['Img','Xnotch1','Ynotch1','Xnotch2','Ynotch2','Xattach','Yattach']) 
    
    # Find landmarks
    for s in StackList:
        nimg = len( io.imread(P + '/' + s + '.tif')) # number of images in the stack

        for i in range(nimg):
            if s not in np.unique(RefPts.index) or (s in np.unique(RefPts.index) and i not in np.unique(RefPts.loc[RefPts.index == s,'Img'].values)) or overwrite :

                ### Identifying landmarks
                # identifying notches manually 
                NotchesRef1 = getPointsCoordinates_alignementNotch(P,s,1,i,'Select the notch 1 - '+ str(s)+ ' - Img ' +str(i)+'/'+str(nimg))
                NotchesRef2 = getPointsCoordinates_alignementNotch(P,s,1,i,'Select the notch 2 - '+ str(s)+ ' - Img ' +str(i)+'/'+str(nimg))
                AttachRef = getPointsCoordinates_alignement_Attach(P,s,1,i,'Select the attach point - '+ str(s)+ ' - Img ' +str(i)+'/'+str(nimg))

                # Save the points that were clicked
                data = {'Img': i,
                        'Xnotch1':NotchesRef1[0][0],
                        'Ynotch1':NotchesRef1[0][1],
                        'Xnotch2':NotchesRef2[0][0],
                        'Ynotch2':NotchesRef2[0][1],
                        'Xattach':AttachRef[0][0],
                        'Yattach':AttachRef[0][1]}            

                RefPts = RefPts.append(pd.DataFrame(data=data,index = [s]))
                
                

                RefPts.to_csv(P + '/clickedpoints_alltimes.csv',index_label = 'Ind')
            
    return RefPts


#%% Rotate and translate all the images to align Notches and attach point

def RotateTranslate(StackList,P, ** kwargs): # add the factor as an option for every gemma

    factor = [0.2]*len(StackList)
    DebugPlots = False # add this possibility
    ImgList = False # add this possibility
    
    for key, value in kwargs.items(): 
        if key == 'DebugPlots':
            DebugPlots = value 
        elif key == 'ImgList':
            ImgList = value 
        elif key == 'factor':
            factor = value 

        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    if not os.path.exists(P + 'Aligned'):
            os.mkdir(P + 'Aligned')
    
    RefPts = pd.read_csv(P + '/clickedpoints_alltimes.csv', index_col = 'Ind')
    
    n_factor = 0
    for s in StackList:
        RGBstack = io.imread(P + '/' + s + '.tif') # get the tiff stack
        nimg = len(RGBstack) # number of images in the stack
        
        RGBstack = cv.copyMakeBorder(
                RGBstack,
                top=int(np.shape(RGBstack)[0]*factor[n_factor]),
                bottom=int(np.shape(RGBstack)[0]*factor[n_factor]),
                left=int(np.shape(RGBstack)[1]*factor[n_factor]),
                right=int(np.shape(RGBstack)[1]*factor[n_factor]),
                borderType=cv.BORDER_CONSTANT,
                value=[0,0,0])
        n_factor += 1
    
        if not os.path.exists(P + '/Aligned/' + s + '_Aligned'):
            os.mkdir(P + '/Aligned/' + s + '_Aligned') # create binarize image folder

        print('Processing ' + s + ' :')    

        for i in range(nimg):
            
            XA = RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i), 'Xattach'].values
            #YA = RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i), 'Yattach'].values

            Xn1 = RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i), 'Xnotch1'].values
            Yn1 = RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i), 'Ynotch1'].values

            Xn2 = RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i), 'Xnotch2'].values
            Yn2 = RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == i), 'Ynotch2'].values
                
            img = RGBstack[i]
            
            # rotation to align notches
            angle = np.arctan((Yn2-Yn1)/abs(Xn2-Xn1))
            new_img = Image.fromarray(img)
            new_img = np.array(new_img.rotate(angle*180/np.pi))
            
            # translation to align attach point
            if i == 0:
                XA_ref = XA
                Yn1_ref = Yn1
                shifted = new_img
            else :
                M = np.float32([[1,0,XA-XA_ref], [0,1,Yn1-Yn1_ref]])
                shifted = cv.warpAffine(new_img, M, (new_img.shape[1],new_img.shape[0])) # enlever le factor
            
            
            io.imsave(P + 'Aligned/' + s + '_Aligned/' + str(i) + '.tif', np.uint8(shifted), plugin='tifffile')
        return
    
    
def RotateTranslate_Automatic(GD, StackList,P, stringName,  ** kwargs): # add the factor as an option for every gemma

    factor = [0.2]*len(StackList)
    DebugPlots = False # add this possibility
    ImgList = False # add this possibility
    
    for key, value in kwargs.items(): 
        if key == 'DebugPlots':
            DebugPlots = value 
        elif key == 'ImgList':
            ImgList = value 
        elif key == 'factor':
            factor = value 

        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    if not os.path.exists(P + '/Aligned'):
        os.mkdir(P + '/Aligned')
        
    if 'XAAl' not in list(GD. columns. values):
        GD['XAAl'] = 0
        GD['YAAl'] = 0
        GD['Xn1Al'] = 0
        GD['Yn1Al'] = 0
        GD['Xn2Al'] = 0
        GD['Yn2Al'] = 0
         
    n_factor = 0

    for s in StackList:
        RGBstack = io.imread(P + '/' + s + '.tif') # get the tiff stack
        nimg = len(RGBstack) # number of images in the stack
        
        
        DontProcess = False
    
        if not os.path.exists(P + '/Aligned/' + s + '_Aligned'):
            os.mkdir(P + '/Aligned/' + s + '_Aligned') # create binarize image folder
        else :
            n_aligned = len([entry for entry in os.listdir(P + '/Aligned/' + s + '_Aligned') if os.path.isfile(os.path.join(P + '/Aligned/' + s + '_Aligned', entry))])
            if nimg == n_aligned :
                DontProcess = True

        print('Processing ' + s + ' :')    
        if not DontProcess : 
            for i in range(nimg):
                
                img = RGBstack[i]
                img = cv.copyMakeBorder(
                        img,
                        top=int(np.shape(img)[0]*factor[n_factor]),
                        bottom=int(np.shape(img)[0]*factor[n_factor]),
                        left=int(np.shape(img)[1]*factor[n_factor]),
                        right=int(np.shape(img)[1]*factor[n_factor]),
                        borderType=cv.BORDER_CONSTANT,
                        value=[0,0,0])
                
    
                XA = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xattach'].values + int(np.shape(RGBstack[i])[1]*factor[n_factor])
                YA = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Yattach'].values + int(np.shape(RGBstack[i])[0]*factor[n_factor])
    
                Xn1 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xnotch1'].values + int(np.shape(RGBstack[i])[1]*factor[n_factor])
                Yn1 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Ynotch1'].values + int(np.shape(RGBstack[i])[0]*factor[n_factor])
    
                Xn2 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Xnotch2'].values + int(np.shape(RGBstack[i])[1]*factor[n_factor])
                Yn2 = GD.loc[(GD.index == s) & (GD['Img'] == i), 'Ynotch2'].values + int(np.shape(RGBstack[i])[0]*factor[n_factor])
                
                GD.loc[(GD.index == s) & (GD['Img'] == i),'Xborder'] = int(np.shape(RGBstack[i])[1]*factor[n_factor])
                GD.loc[(GD.index == s) & (GD['Img'] == i),'Yborder'] = int(np.shape(RGBstack[i])[0]*factor[n_factor])
                
                ## align based on the notches position
                
                # rotation to align notches
                angle = np.arctan(abs(Yn2-Yn1)/abs(Xn2-Xn1))*180/np.pi
                angle = angle[0]
                
                if np.argmin([Xn1,Xn2]) == np.argmin([Yn1,Yn2]):
                    direction = 1
                    Xwhich = np.argmin([Xn1,Xn2])
                    center=(int([Xn1,Xn2][Xwhich]), int([Yn1,Yn2][Xwhich]))
                    transfoMatrix = cv.getRotationMatrix2D(center=(int([Xn1,Xn2][Xwhich]), int([Yn1,Yn2][Xwhich])), angle = direction*angle, scale = 1)
                    
                elif np.argmin([Xn1,Xn2]) != np.argmin([Yn1,Yn2]):
                    direction = -1
                    Xwhich = np.argmax([Xn1,Xn2])
                    center=(int([Xn1,Xn2][Xwhich]), int([Yn1,Yn2][Xwhich]))
                    transfoMatrix = cv.getRotationMatrix2D(center=(int([Xn1,Xn2][Xwhich]), int([Yn1,Yn2][Xwhich])), angle = direction*angle, scale = 1)
                    
                rot_img = cv.warpAffine(img, transfoMatrix, None)# (img.shape[1],img.shape[0]))
                
                # calculate new coordinate for the reference points
                #print(direction) # issue with decision king of direction ?s
                
                Xn1rot, Yn1rot = rotate(center, (Xn1,Yn1), -direction*angle*np.pi/180)
                Xn2rot, Yn2rot = rotate(center, (Xn2,Yn2), -direction*angle*np.pi/180)
                XArot, YArot = rotate(center, (XA,YA), -direction*angle*np.pi/180)
                
        
                
                if YArot < Yn1rot:
                    transfoMatrix = cv.getRotationMatrix2D(center=(int(abs(Xn1rot-Xn2rot)/2+min(Xn1rot,Xn2rot)), int(abs(Yn1rot-Yn2rot)/2+min(Yn1rot,Yn2rot))), angle = direction*180, scale = 1)
                    flip_img = cv.warpAffine(rot_img, transfoMatrix, None)
                    
                    center=(int(abs(Xn1rot-Xn2rot)/2+min(Xn1rot,Xn2rot)), int(abs(Yn1rot-Yn2rot)/2+min(Yn1rot,Yn2rot)))
                  
                    Xrot = abs(Xn1rot-Xn2rot)/2+min(Xn1rot,Xn2rot)
                    Yrot = abs(Yn1rot-Yn2rot)/2+min(Yn1rot,Yn2rot)
                    
                    Xn1Al, Yn1Al = rotate(center, (Xn1rot,Yn1rot), -direction*np.pi)
                    Xn2Al, Yn2Al = rotate(center, (Xn2rot,Yn2rot), -direction*np.pi)
                    XAAl, YAAl = rotate(center, (XArot,YArot), -direction*np.pi)
                    
    
                else :
                    flip_img = rot_img
                    Xrot = abs(Xn1rot-Xn2rot)/2+min(Xn1rot,Xn2rot)
                    Yrot = abs(Yn1rot-Yn2rot)/2+min(Yn1rot,Yn2rot)
                    
                    Xn1Al = Xn1rot
                    Yn1Al= Yn1rot
                    
                    Xn2Al = Xn2rot
                    Yn2Al = Yn2rot
                    
                    XAAl = XArot
                    YAAl = YArot
                
                # translation to align notch
                if i == 0:
                    X_ref = Xrot
                    Y_ref = Yrot
                    shifted = flip_img
                else :
                    M = np.float32([[1,0,X_ref-Xrot], [0,1,Y_ref-Yrot]])
                    shifted = cv.warpAffine(flip_img, M, None) # enlever le factor
                  
                # NB il faut pré-créer les colonnes sinon première propagule = nan
                GD.loc[(GD.index == s) & (GD['Img'] == i),'XAAl'] = XAAl- Xrot + X_ref
                GD.loc[(GD.index == s) & (GD['Img'] == i),'YAAl'] = YAAl - Yrot + Y_ref
                
                GD.loc[(GD.index == s) & (GD['Img'] == i),'Xn1Al'] = Xn1Al- Xrot + X_ref
                GD.loc[(GD.index == s) & (GD['Img'] == i),'Yn1Al'] = Yn1Al - Yrot + Y_ref
                
                GD.loc[(GD.index == s) & (GD['Img'] == i),'Xn2Al'] = Xn2Al- Xrot + X_ref
                GD.loc[(GD.index == s) & (GD['Img'] == i),'Yn2Al'] = Yn2Al - Yrot + Y_ref
                
                if DebugPlots :
                    fig, [[ax0, ax1], [ax3, ax4]] = plt.subplots(nrows = 2, ncols=2,dpi = 500)    
                    fig.suptitle('Debug plots for "Rotate" function :')
                    ax0.imshow(img, cmap='gray')
                    ax0.scatter(XA,YA, s =10, c = 'r')
                    ax0.scatter(Xn1,Yn1, s =10, c = 'b')
                    ax0.scatter(Xn2,Yn2, s =10, c = 'g')
                    ax0.set_title("Image + Border")
                    ax0.axis('off')
                    
                    ax1.imshow(rot_img, cmap='gray')
                    ax1.scatter(XArot,YArot, s =10, c = 'r')
                    ax1.scatter(Xn1rot,Yn1rot, s =10, c = 'b')
                    ax1.scatter(Xn2rot,Yn2rot, s =10, c = 'g')
                    ax1.set_title("Rotate")
                    ax1.axis('off')       
                    fig.tight_layout()
                    
                    ax3.imshow(flip_img, cmap='gray')
                    ax3.scatter(XAAl,YAAl, s =10, c = 'r')
                    ax3.scatter(Xn1Al,Yn1Al, s =10, c = 'b')
                    ax3.scatter(Xn2Al,Yn2Al, s =10, c = 'g')
                    ax3.set_title("Flip")
                    ax3.axis('off')
                    
                    ax4.imshow(shifted, cmap='gray')
                    ax4.set_title("Aligned")
                    ax4.scatter(XAAl- Xrot + X_ref,YAAl- Yrot + Y_ref, s =10, c = 'r')
                    ax4.scatter(Xn1Al- Xrot + X_ref,Yn1Al- Yrot + Y_ref, s =10, c = 'b')
                    ax4.scatter(Xn2Al- Xrot + X_ref,Yn2Al- Yrot + Y_ref, s =10, c = 'g')
                    ax4.axis('off')        
                    fig.tight_layout()
                    plt.show()
            
            
            
                io.imsave(P + '/Aligned/' + s + '_Aligned/' + str(i) + '.tif', np.uint8(shifted), plugin='tifffile')
                GD.to_csv(P + '/GlobalData' + stringName + '_Landmarks_aligned.csv',index_label = 'Ind')
        n_factor += 1
    
    return
    
            
            

#%% Remove background from the aligned images

def RemoveBackgroundStack(StackList, P, Scale, **kwargs):

    #init and read kwargs
    DebugPlots = False                            
    HSVmin = (30, 20, 70)
    HSVmax = (60, 120,220)
    ImgList = [0, 20, 40]
    saveWB = False
    factor1 = 2
    factor2 = 30
    Areath = 5*1e3
    
    for key, value in kwargs.items(): 
        if key == 'DebugPlots':
            DebugPlots = value 
        elif key == 'ImgList':
            ImgList = value 
        elif key == 'saveWB':
            saveWB = value 
        elif key == 'HSVrange':
            HSVmin = value[0]
            HSVmax = value[1]
        elif key == "Binfactor1":
            factor1 = value
        elif key == "Binfactor2":
            factor2 = value
        elif key == "Binthreshold" :
            Areath = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
     

    # Savefolders for WB and Binary stacks
    if saveWB:
        if not os.path.exists(P + '/WhiteBalanced'):
            os.mkdir(P + '/WhiteBalanced') # create global folder 

    if not os.path.exists(P + '/Processed'):
        os.mkdir(P + '/Processed') # create global folder 
            
            
    if os.path.exists(P + '/WhiteBalanceData.csv'):
        WBsaving = pd.read_csv(P + '/WhiteBalanceData.csv', index_col = 'Ind')
    else:                
        WBsaving = pd.DataFrame(data=None,columns=['lum','whiteR','whiteG','whiteB']) 
        
    
    
    for s in StackList:
        
        if not os.path.exists(P + '/Processed/' + s + '_NoBackground'):
            # shutil.rmtree(P + '/' + s + '_NoBackground') # remove folder and contents
            os.mkdir(P + '/Processed/' + s + '_NoBackground') # create binarize image folder
            isBin = False
        else:
            isBin = True
            
        print('Processing ' + s + ' :')  
        
        ProcessedPath = P + '/Aligned/' + str(s) + '_Aligned'
        
        n_img = len([entry for entry in os.listdir(ProcessedPath) if os.path.isfile(os.path.join(ProcessedPath, entry))])

        RGBstack = np.asarray([io.imread(ProcessedPath + '/' + str(i) + '.tif') for i in range(n_img)])# get the tiff stack
        Bckp_RGBstack = np.asarray([io.imread(ProcessedPath + '/' + str(i) + '.tif') for i in range(n_img)]) # get the tiff stack for comparison
        
        if DebugPlots:            
            imglist = ImgList
        else:
            imglist = range(n_img)  
            #imglist = range(49)  
            
        #### Adjusting brightness
        # getting reference value
        Img0 = io.imread(ProcessedPath + '/' + str(0) + '.tif')
        
        HSVref = cv.cvtColor(Img0, cv.COLOR_BGR2HSV)
        Brightness = HSVref[:,:,2]
        RefBright = Brightness.mean()

        # harmonizing between images
        for i in imglist:
            
            HSVslice = cv.cvtColor(RGBstack[i,:,:,:], cv.COLOR_BGR2HSV)
        
            Brightness = HSVslice[:,:,2].astype(float)
            Bright = Brightness.mean()
            factor = RefBright / Bright
            CorrecBright = Brightness*factor
            CorrecBright = np.round(CorrecBright)
            CorrecBright[CorrecBright>254] = 255
            newfactor = RefBright / CorrecBright.mean()
            TrueCorrecBright = CorrecBright*newfactor
            TrueCorrecBright = np.round(TrueCorrecBright)
            TrueCorrecBright[TrueCorrecBright>254] = 255
            HSVslice[:,:,2] = TrueCorrecBright.astype('uint8')
            RGBstack[i,:,:,:] = cv.cvtColor(HSVslice, cv.COLOR_HSV2BGR)
            
            
        #### adjusting white balance
        if WBsaving.loc[(WBsaving.index == s)].empty:            
            # ask user to click
            get_ipython().run_line_magic('matplotlib', 'qt')
            plt.figure(dpi=250)
            plt.title('Select an area for white balance')
            plt.imshow(RGBstack[0])
            pts = np.asarray(plt.ginput(-1, timeout=-1))
            plt.close()
            get_ipython().run_line_magic('matplotlib', 'inline')
            
            
            # indicices of convex hull of set of points pts
            CH = ConvexHull(pts)

            # polygon defining the hull
            p = mpl.path.Path(pts[CH.vertices,:])

            # set of image pixels coordinates
            xx,yy = np.meshgrid(np.linspace(0,np.shape(RGBstack[0])[1],num=np.shape(RGBstack[0])[1],endpoint=False),
                                np.linspace(0,np.shape(RGBstack[0])[0],num=np.shape(RGBstack[0])[0],endpoint=False))

            # mask for the part of the image to use for white balance
            mask = np.reshape(p.contains_points(np.hstack((xx.flatten()[:,np.newaxis],yy.flatten()[:,np.newaxis]))),
                              np.shape(RGBstack[0])[0:2])

            # balancing white
            whiteR = RGBstack[0][mask,0].mean()
            whiteG = RGBstack[0][mask,1].mean()
            whiteB = RGBstack[0][mask,2].mean()
            lum = (whiteR + whiteG + whiteB)/3
            
            # Save the points that were clicked
            data = {'lum': lum,
                    'whiteR': whiteR,
                    'whiteG': whiteG,
                    'whiteB': whiteB}         
            
            WBsaving = WBsaving.append(pd.DataFrame(data=data,index = [s]))

            WBsaving.to_csv(P + '/WhiteBalanceData.csv',index_label = 'Ind')
            
        else:
            whiteR = WBsaving.loc[s,'whiteR']
            whiteG = WBsaving.loc[s,'whiteG']
            whiteB = WBsaving.loc[s,'whiteB']
            lum = WBsaving.loc[s,'lum']


            
        # adjusting stack white balance
        RGBstack[:,:,:,0] = RGBstack[:,:,:,0] * lum / whiteR
        RGBstack[:,:,:,1] = RGBstack[:,:,:,1] * lum / whiteG
        RGBstack[:,:,:,2] = RGBstack[:,:,:,2] * lum / whiteB

        if saveWB:        
            tifff.imsave(P + '/WhiteBalanced/' + s + '.tif', RGBstack)
            


        
        
        for i in imglist:
            
            Img = RGBstack[i]
            
            if DebugPlots&0:
                fig,ax = plt.subplots(ncols=2,dpi=200)
                ax[0].set_title('Original image')
                ax[0].imshow(Bckp_RGBstack[i])
                ax[1].set_title('White balanced image')
                ax[1].imshow(Img)
                
                plt.show()
            
            # Binarization      
            if not isBin:
                print('Binarization of image ' + str(i+1) + '/' + str(len(RGBstack)).ljust(15), flush=True, end = '\r')
                
                BinImg = RemoveBackground(Img,Scale,HSVmin,HSVmax,debug =DebugPlots, Binfactor1 = factor1, Binfactor2 = factor2,  Binthreshold = Areath)
                
                io.imsave(P + '/Processed/' + s + '_NoBackground/' + str(i) + '.tif', np.uint8(BinImg*255), plugin='tifffile')
            
            else:
                if os.path.exists(P + '/Processed/' + s + '_NoBackground/' + str(i) + '.tif'):
                    BinImg = io.imread(P + '/Processed/' + s + '_NoBackground/' + str(i) + '.tif') 
                else:
                    print('Binarization of image ' + str(i+1) + '/' + str(len(RGBstack)).ljust(15), flush=True, end = '\r')
                    
                    BinImg = RemoveBackground(Img,Scale,HSVmin,HSVmax,debug =DebugPlots, Binfactor1 = factor1, Binfactor2 = factor2, Binthreshold = Areath)
                    
                    io.imsave(P + '/Processed/' + s + '_NoBackground/' + str(i) + '.tif', np.uint8(BinImg*255), plugin='tifffile')
            


        print('Binarization done for ' + str(len(RGBstack)) + '/' + str(len(RGBstack)) + ' images'.ljust(20), flush=True)

        
    return


def RemoveBackgroundStack_Flow(StackList, P, Scale, **kwargs):

    #init and read kwargs
    DebugPlots = False                            
    HSVmin = (30, 20, 70)
    HSVmax = (60, 120,220)
    ImgList = [0, 20, 40]
    saveWB = False
    factor1 = 2
    factor2 = 30
    Areath = 5*1e3
    DebugPlotsFlow = False
    
    for key, value in kwargs.items(): 
        if key == 'DebugPlots':
            DebugPlots = value 
        elif key == 'ImgList':
            ImgList = value 
        elif key == 'saveWB':
            saveWB = value 
        elif key == 'HSVrange':
            HSVmin = value[0]
            HSVmax = value[1]
        elif key == "Binfactor1":
            factor1 = value
        elif key == "Binfactor2":
            factor2 = value
        elif key == "Binthreshold" :
            Areath = value
        elif key == 'DebugPlotsFlow':
            DebugPlotsFlow = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
     

    # Savefolders for WB and Binary stacks
    if saveWB:
        if not os.path.exists(P + '/WhiteBalanced'):
            os.mkdir(P + '/WhiteBalanced') # create global folder 

    if not os.path.exists(P + '/Processed'):
        os.mkdir(P + '/Processed') # create global folder 
            
            
    if os.path.exists(P + '/WhiteBalanceData.csv'):
        WBsaving = pd.read_csv(P + '/WhiteBalanceData.csv', index_col = 'Ind')
    else:                
        WBsaving = pd.DataFrame(data=None,columns=['lum','whiteR','whiteG','whiteB']) 
        
    
    
    for s in StackList:
        
        if not os.path.exists(P + '/Flow/' + s + '_FlowNoBackground'):
            os.mkdir(P + '/Flow/' + s + '_FlowNoBackground') # create binarize image folder
            isBin = False
        else:
            isBin = True
            
        print('Processing ' + s + ' :')  
        
        OriginalPath = P + '/Aligned/' + str(s) + '_Aligned'
        FlowPath = P + '/Flow/' + str(s) + '_Flow'
        
        n_img = int(len([entry for entry in os.listdir(FlowPath) if os.path.isfile(os.path.join(FlowPath, entry))])/2)

        RGBstack = np.asarray([io.imread(OriginalPath + '/' + str(i) + '.tif') for i in range(1,n_img+1)])# get the tiff stack
        #Bckp_RGBstack = np.asarray([io.imread(OriginalPath + '/' + str(i) + '.tif') for i in range(1,n_img+1)])# get the tiff stack
        Flowstack_U = np.asarray([io.imread(FlowPath + '/' + str(i) + '_U.tif') for i in range(1,n_img+1)]) 
        Flowstack_V = np.asarray([io.imread(FlowPath + '/' + str(i) + '_V.tif') for i in range(1,n_img+1)])
        
        if DebugPlots:            
            imglist = ImgList
        else:
            imglist = range(n_img)  
            #imglist = range(49)  
            
        #### Adjusting brightness
        # getting reference value
        Img0 = io.imread(OriginalPath + '/' + str(0) + '.tif')
        
        HSVref = cv.cvtColor(Img0, cv.COLOR_BGR2HSV)
        Brightness = HSVref[:,:,2]
        RefBright = Brightness.mean()

        # harmonizing between images
        for i in imglist:
            
            HSVslice = cv.cvtColor(RGBstack[i,:,:,:], cv.COLOR_BGR2HSV)
        
            Brightness = HSVslice[:,:,2].astype(float)
            Bright = Brightness.mean()
            factor = RefBright / Bright
            CorrecBright = Brightness*factor
            CorrecBright = np.round(CorrecBright)
            CorrecBright[CorrecBright>254] = 255
            newfactor = RefBright / CorrecBright.mean()
            TrueCorrecBright = CorrecBright*newfactor
            TrueCorrecBright = np.round(TrueCorrecBright)
            TrueCorrecBright[TrueCorrecBright>254] = 255
            HSVslice[:,:,2] = TrueCorrecBright.astype('uint8')
            RGBstack[i,:,:,:] = cv.cvtColor(HSVslice, cv.COLOR_HSV2BGR)
            
            
        #### adjusting white balance
        if WBsaving.loc[(WBsaving.index == s)].empty:            
            # ask user to click
            get_ipython().run_line_magic('matplotlib', 'qt')
            plt.figure(dpi=250)
            plt.title('Select an area for white balance')
            plt.imshow(RGBstack[0])
            pts = np.asarray(plt.ginput(-1, timeout=-1))
            plt.close()
            get_ipython().run_line_magic('matplotlib', 'inline')
            
            
            # indicices of convex hull of set of points pts
            CH = ConvexHull(pts)

            # polygon defining the hull
            p = mpl.path.Path(pts[CH.vertices,:])

            # set of image pixels coordinates
            xx,yy = np.meshgrid(np.linspace(0,np.shape(RGBstack[0])[1],num=np.shape(RGBstack[0])[1],endpoint=False),
                                np.linspace(0,np.shape(RGBstack[0])[0],num=np.shape(RGBstack[0])[0],endpoint=False))

            # mask for the part of the image to use for white balance
            mask = np.reshape(p.contains_points(np.hstack((xx.flatten()[:,np.newaxis],yy.flatten()[:,np.newaxis]))),
                              np.shape(RGBstack[0])[0:2])

            # balancing white
            whiteR = RGBstack[0][mask,0].mean()
            whiteG = RGBstack[0][mask,1].mean()
            whiteB = RGBstack[0][mask,2].mean()
            lum = (whiteR + whiteG + whiteB)/3
            
            # Save the points that were clicked
            data = {'lum': lum,
                    'whiteR': whiteR,
                    'whiteG': whiteG,
                    'whiteB': whiteB}         
            
            WBsaving = WBsaving.append(pd.DataFrame(data=data,index = [s]))

            WBsaving.to_csv(P + '/WhiteBalanceData.csv',index_label = 'Ind')
            
        else:
            whiteR = WBsaving.loc[s,'whiteR']
            whiteG = WBsaving.loc[s,'whiteG']
            whiteB = WBsaving.loc[s,'whiteB']
            lum = WBsaving.loc[s,'lum']


            
        # adjusting stack white balance
        RGBstack[:,:,:,0] = RGBstack[:,:,:,0] * lum / whiteR
        RGBstack[:,:,:,1] = RGBstack[:,:,:,1] * lum / whiteG
        RGBstack[:,:,:,2] = RGBstack[:,:,:,2] * lum / whiteB

        if saveWB:        
            tifff.imsave(P + '/WhiteBalanced/' + s + '.tif', RGBstack)
            


        
        
        for i in imglist:
            
            Img = RGBstack[i]
            FlowImg_U = Flowstack_U[i]
            FlowImg_V = Flowstack_V[i]
            
            OriImg = np.copy(FlowImg_U)
            
           
            
            # Binarization      
            if not isBin:
                print('Binarization of image ' + str(i+1) + '/' + str(len(RGBstack)).ljust(15), flush=True, end = '\r')
                
                BinImg_U = RemoveBackground_Flow(Img,FlowImg_U, Scale,HSVmin,HSVmax,debug =DebugPlots, Binfactor1 = factor1, Binfactor2 = factor2,  Binthreshold = Areath)
                BinImg_V = RemoveBackground_Flow(Img,FlowImg_V, Scale,HSVmin,HSVmax,debug =DebugPlots, Binfactor1 = factor1, Binfactor2 = factor2,  Binthreshold = Areath)
                
                io.imsave(P + '/Flow/' + s + '_FlowNoBackground/' + str(i+1) + '_U.tif', BinImg_U, plugin='tifffile')
                io.imsave(P + '/Flow/' + s + '_FlowNoBackground/' + str(i+1) + '_V.tif', BinImg_V, plugin='tifffile')
            
            else:
                if os.path.exists(P + '/Flow/' + s + '_FlowNoBackground/' + str(i+1) + '_U.tif') and os.path.exists(P + '/Flow/' + s + '_NoBackground/' + str(i) + '_V.tif'):
                    BinImg_U = io.imread(P + '/Flow/' + s + '_FlowNoBackground/' + str(i+1) + '_U.tif') 
                    BinImg_V = io.imread(P + '/Flow/' + s + '_FlowNoBackground/' + str(i+1) + '_V.tif') 
                else:
                    print('Binarization of image ' + str(i+1) + '/' + str(len(RGBstack)).ljust(15), flush=True, end = '\r')
                    
                    BinImg_U = RemoveBackground_Flow(Img,FlowImg_U, Scale,HSVmin,HSVmax,debug =DebugPlots, Binfactor1 = factor1, Binfactor2 = factor2,  Binthreshold = Areath)
                    BinImg_V = RemoveBackground_Flow(Img,FlowImg_V, Scale,HSVmin,HSVmax,debug =DebugPlots, Binfactor1 = factor1, Binfactor2 = factor2,  Binthreshold = Areath)
                    
                    io.imsave(P + '/Flow/' + s + '_FlowNoBackground/' + str(i+1) + '_U.tif', BinImg_U, plugin='tifffile')
                    io.imsave(P + '/Flow/' + s + '_FlowNoBackground/' + str(i+1) + '_V.tif', BinImg_V, plugin='tifffile')
                    
            if DebugPlotsFlow : #DebugPlots&0:
                 fig,ax = plt.subplots(ncols=2,dpi=200)
                 ax[0].set_title('Original image')
                 figU = ax[0].imshow(OriImg )
                 fig.colorbar(figU, ax = ax[0])
                 ax[1].set_title('Bin Image')
                 figBin = ax[1].imshow(BinImg_U )
                 fig.colorbar(figBin, ax = ax[1])
                 
                 plt.show()
            


        print('Binarization done for ' + str(len(RGBstack)) + '/' + str(len(RGBstack)) + ' images'.ljust(20), flush=True)

        
    return

def RemoveBackground_Flow(Img, FlowImg, Scale, HSVmin, HSVmax, **kwargs):

    #init and read kwargs
    DebugPlots = False
    factor1 = 2
    factor2 = 30
    Areath = 5*1e3
    
    for key, value in kwargs.items(): 
        if key == 'debug':
            DebugPlots = value
        elif key == "Binfactor1":
            factor1 = value
        elif key == "Binfactor2":
            factor2 = value
        elif key == "Binthreshold" :
            Areath = value
            

    
    hsv = cv.cvtColor(Img, cv.COLOR_RGB2HSV)

    ## mask 
    mask = cv.inRange(hsv, HSVmin, HSVmax)

    ## slice 
    BWimg = mask>0
    
    size1 = np.round(factor1*Scale) # 5µm in pixels
    #size = np.round(4*Scale) # 5µm in pixels for 500 mM shocks and Pase experiment
    
    selem = create_circular_mask(size1,size1) # create circular element for opening
    #selem = diamond(size1) #for image on grid"
    #selem = np.zeros((int(size1)+2,int(size1)+2),  dtype=np.uint8)
    #for ci in range(int(size1)):
     #   selem[ci+1, int(size1)-ci] = 1
    
    
    DilBWimg = binary_closing(BWimg,selem) # image closing
    
    FilledBWimg = remove_small_holes(DilBWimg, area_threshold=Areath) # fills dark regions
    
    Size2 = np.round(factor2*Scale) # 30µm in pixels
    
    Selem = create_circular_mask(Size2,Size2) # create circular element for opening

    FinalImg = binary_opening(FilledBWimg,Selem) # image opening
    

    
    #Keep only the biggest gemmae
    FinalImg = KeepMainGemma(FinalImg*255)
    
    FlowImg[np.invert(FinalImg)] =  np.nan
    Img_nobackground = FlowImg
    
    #Img_nobackground = np.multiply(FlowImg,np.repeat(FinalImg[:, :, np.newaxis], 3, axis=2))*255 # Img[FinalImg]#np.where(FinalImg, Img, Img*0)
    
    if DebugPlots:       
        print('In debug')
       
        DilBWonImg = np.multiply(Img,np.repeat(FilledBWimg[:, :, np.newaxis], 3, axis=2))
        FinalOnImg = np.multiply(Img,np.repeat(FinalImg[:, :, np.newaxis], 3, axis=2))
        BWonImgInv =np.multiply(Img,np.repeat(np.invert(BWimg[:, :, np.newaxis]), 3, axis=2))
        DilBWonImgInv = np.multiply(Img,np.repeat(np.invert(FilledBWimg[:, :, np.newaxis]), 3, axis=2))
        FinalOnImgInv = np.multiply(Img,np.repeat(np.invert(FinalImg[:, :, np.newaxis]), 3, axis=2))
        
        fig, [[ax0, ax1, ax2], [ax3, ax4, ax5]] = plt.subplots(nrows = 2, ncols=3,dpi = 500)    
        fig.suptitle('Debug plots for "Binarize" function :')
        ax0.imshow(Img, cmap='gray')
        ax0.set_title("Original & Thresholded images")
        ax0.axis('off')
        ax3.imshow(BWonImgInv, cmap='gray')
        ax3.axis('off')
        ax1.imshow(DilBWonImgInv, cmap='gray')
        ax1.set_title("Closed image")
        ax1.axis('off') 
        ax2.imshow(FinalOnImgInv, cmap='gray')
        ax2.set_title("Final mask")
        ax2.axis('off')        
        fig.tight_layout()
        ax4.imshow(DilBWonImg, cmap='gray')
        ax4.axis('off') 
        ax5.imshow(FinalOnImg, cmap='gray')
        ax5.axis('off')        
        fig.tight_layout()
        plt.show()
        
        print('')
    
    return(Img_nobackground)


def RemoveBackground(Img, Scale, HSVmin, HSVmax, **kwargs):

    #init and read kwargs
    DebugPlots = False
    factor1 = 2
    factor2 = 30
    Areath = 5*1e3
    
    for key, value in kwargs.items(): 
        if key == 'debug':
            DebugPlots = value
        elif key == "Binfactor1":
            factor1 = value
        elif key == "Binfactor2":
            factor2 = value
        elif key == "Binthreshold" :
            Areath = value
            

    
    hsv = cv.cvtColor(Img, cv.COLOR_RGB2HSV)

    ## mask 
    mask = cv.inRange(hsv, HSVmin, HSVmax)

    ## slice 
    BWimg = mask>0
    
    size1 = np.round(factor1*Scale) # 5µm in pixels
    #size = np.round(4*Scale) # 5µm in pixels for 500 mM shocks and Pase experiment
    
    selem = create_circular_mask(size1,size1) # create circular element for opening
    #selem = diamond(size1) #for image on grid"
    #selem = np.zeros((int(size1)+2,int(size1)+2),  dtype=np.uint8)
    #for ci in range(int(size1)):
     #   selem[ci+1, int(size1)-ci] = 1
    
    
    DilBWimg = binary_closing(BWimg,selem) # image closing
    
    FilledBWimg = remove_small_holes(DilBWimg, area_threshold=Areath) # fills dark regions
    
    Size2 = np.round(factor2*Scale) # 30µm in pixels
    
    Selem = create_circular_mask(Size2,Size2) # create circular element for opening

    FinalImg = binary_opening(FilledBWimg,Selem) # image opening
    

    
    #Keep only the biggest gemmae
    FinalImg = KeepMainGemma(FinalImg*255)
    
    Img_nobackground = np.multiply(Img,np.repeat(FinalImg[:, :, np.newaxis], 3, axis=2))*255 # Img[FinalImg]#np.where(FinalImg, Img, Img*0)
    
    if DebugPlots:       
        print('In debug')
       
        DilBWonImg = np.multiply(Img,np.repeat(FilledBWimg[:, :, np.newaxis], 3, axis=2))
        FinalOnImg = np.multiply(Img,np.repeat(FinalImg[:, :, np.newaxis], 3, axis=2))
        BWonImgInv =np.multiply(Img,np.repeat(np.invert(BWimg[:, :, np.newaxis]), 3, axis=2))
        DilBWonImgInv = np.multiply(Img,np.repeat(np.invert(FilledBWimg[:, :, np.newaxis]), 3, axis=2))
        FinalOnImgInv = np.multiply(Img,np.repeat(np.invert(FinalImg[:, :, np.newaxis]), 3, axis=2))
        
        fig, [[ax0, ax1, ax2], [ax3, ax4, ax5]] = plt.subplots(nrows = 2, ncols=3,dpi = 500)    
        fig.suptitle('Debug plots for "Binarize" function :')
        ax0.imshow(Img, cmap='gray')
        ax0.set_title("Original & Thresholded images")
        ax0.axis('off')
        ax3.imshow(BWonImgInv, cmap='gray')
        ax3.axis('off')
        ax1.imshow(DilBWonImgInv, cmap='gray')
        ax1.set_title("Closed image")
        ax1.axis('off') 
        ax2.imshow(FinalOnImgInv, cmap='gray')
        ax2.set_title("Final mask")
        ax2.axis('off')        
        fig.tight_layout()
        ax4.imshow(DilBWonImg, cmap='gray')
        ax4.axis('off') 
        ax5.imshow(FinalOnImg, cmap='gray')
        ax5.axis('off')        
        fig.tight_layout()
        plt.show()
        
        print('')
    
    return(Img_nobackground)


def KeepMainGemma(BinImg):      
    
    BinImg = BinImg.astype(np.uint8)
    cnts, _ = cv.findContours(BinImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if np.shape(cnts)[0] > 1: 
        cnts_area = [cv.contourArea(c) for c in cnts]
        cntMax = np.argmax(cnts_area)
        for i in range(np.shape(cnts)[0]):
            if i != cntMax:
                c = cnts[i]
                mask = np.ones(BinImg.shape[:2], dtype="uint8") * 255
                cv.drawContours(mask, [c], -1, 0, -1)
                BinImg = cv.bitwise_and(BinImg, BinImg, mask=mask)
    return BinImg > 0




#%% Calculate the optical flow (from avi format)


def MainOpticalFlow_AutomaticLandmarks_RemoveBgLast(P, StackList, Scale, ToDo, stringName, **kwargs):
    
    # keyword arguments to find reference point
    overwriteRefs = False
    
    # keyword arguments for rotate/translate image
    factorA = [0.2]*len(StackList)
    DebugPlotsA = False
    
    
    ImgList = [0, 20, 40]
    
    # keyword argument for remove background
    DebugPlotsB = False                            
    HSVmin = (30, 20, 70)
    HSVmax = (60, 120,220)
    saveWB = False
    factor1 = 2
    factor2 = 30
    Areath = 5*1e3
    DebugPlotsFlow = False
    
    # keyword arguments for optical flow
    to_gray = True
    
    for key, value in kwargs.items(): 
        if key == 'overwriteRefs':
            overwriteRefs = value
        elif key == 'factorA':
            factorA = value
        elif key == 'DebugPlotsA':
            DebugPlotsA = value
        elif key == 'ImgList':
            ImgList= value
        elif key == 'DebugPlotsB':
            DebugPlotsB = value
        elif key == 'saveWB':
            saveWB = value 
        elif key == 'HSVrange':
            HSVmin = value[0]
            HSVmax = value[1]
        elif key == "Binfactor1":
            factor1 = value
        elif key == "Binfactor2":
            factor2 = value
        elif key == "Binthreshold" :
            Areath = value
        elif key == "to_gray":
            to_gray = value
        elif key == 'DebugPlotsFlow':
            DebugPlotsFlow = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    DoSelect = False
    DoAlign = False
    DoBackground = False
    DoOpticalFlow = False
    
    if "S" in ToDo:
        DoSelect = True
    if "A" in ToDo:
        DoAlign = True
    if "B" in ToDo:
        DoBackground = True
    if "O" in ToDo :
        DoOpticalFlow = True
    
    if DoSelect:
        GlobalData = pd.read_csv(P + '/GlobalData' + stringName + '_AreaFit.csv', index_col = 'Ind')
        StackList = [s for s in StackList if (s in np.unique(GlobalData.index.values)) ]
        OpticalStack, Not2notches, Confined = selectGemmae(P, StackList, size = 300, nimg = 10)
        
        np.savetxt(P+"/OpticalStack.csv",OpticalStack,delimiter =", ",fmt ='% s')
        np.savetxt(P+"/Not2notchesStack.csv",Not2notches,delimiter =", ",fmt ='% s')
        np.savetxt(P+"/ConfinedStack.csv",Confined,delimiter =", ",fmt ='% s')
    
    else :
        OpticalStack = np.loadtxt(P+"/OpticalStack.csv", delimiter =", ", dtype = str)
        print(OpticalStack)
    
    if DoAlign:
        
        if not os.path.exists(P + '/GlobalData' + stringName + '_Landmarks.csv'):
            if not os.path.exists(P + '/GlobalData' + stringName + '_Landmarks_tmp.csv'):
                ContourData = pd.read_csv(P + '/ContourData' + stringName + '_AreaFit.csv', index_col = 'Ind')
                GlobalData = pd.read_csv(P + '/GlobalData' + stringName + '_AreaFit.csv', index_col = 'Ind')
                print('\n Loaded AreaFit file.')
            else:            
                ContourData = pd.read_csv(P + '/ContourData' + stringName + '_Landmarks_tmp.csv', index_col = 'Ind')
                GlobalData = pd.read_csv(P + '/GlobalData' + stringName + '_Landmarks_tmp.csv', index_col = 'Ind')
                print('\n Loaded Landmarks_tmp file.')
        else:            
            ContourData = pd.read_csv(P + '/ContourData' + stringName + '_Landmarks.csv', index_col = 'Ind')
            GlobalData = pd.read_csv(P + '/GlobalData' + stringName + '_Landmarks.csv', index_col = 'Ind')
            print('\n Loaded Landmarks file.')
        
        ContourData_LM, GlobalData_LM =getLandmarks(ContourData,GlobalData,OpticalStack,Scale,P,stringName, FirstSlice=False, Overwrite = overwriteRefs)
        
        if os.path.exists(P + '/GlobalData' + stringName + '_Landmarks_aligned.csv'):
            GlobalData_LM = pd.read_csv(P + '/GlobalData' + stringName + '_Landmarks_aligned.csv', index_col = 'Ind')
            
        RotateTranslate_Automatic(GlobalData_LM, OpticalStack,P, stringName, factor = factorA, DebugPlots = DebugPlotsA, ImgList = ImgList)
        
        
    if DoOpticalFlow:
    
        if not os.path.exists(P + '/Flow'):
                os.mkdir(P + '/Flow') 
    
        for s in OpticalStack:
    
            print('Processing ' + s + ' :')  
            
            if not os.path.exists(P + '/Flow/' + s + '_Flow'):
                os.mkdir(P + '/Flow/' + s + '_Flow')
            
            OpticalFlow_Bckg(P, s, to_gray=to_gray)
            
    if DoBackground:
        RemoveBackgroundStack_Flow(OpticalStack, P, Scale, DebugPlots = DebugPlotsB, ImgList = ImgList,saveWB = saveWB, HSVrange = [HSVmin, HSVmax], Binfactor1 = factor1, Binfactor2 = factor2, Binthreshold = Areath, DebugPlotsFlow = DebugPlotsFlow)
    
    return

def MainOpticalFlow_AutomaticLandmarks(P, StackList, Scale, ToDo, stringName, **kwargs):
    
    # keyword arguments to find reference point
    overwriteRefs = False
    
    # keyword arguments for rotate/translate image
    factorA = [0.2]*len(StackList)
    DebugPlotsA = False
    
    
    ImgList = [0, 20, 40]
    
    # keyword argument for remove background
    DebugPlotsB = False                            
    HSVmin = (30, 20, 70)
    HSVmax = (60, 120,220)
    saveWB = False
    factor1 = 2
    factor2 = 30
    Areath = 5*1e3
    
    # keyword arguments for optical flow
    to_gray = True
    
    for key, value in kwargs.items(): 
        if key == 'overwriteRefs':
            overwriteRefs = value
        elif key == 'factorA':
            factorA = value
        elif key == 'DebugPlotsA':
            DebugPlotsA = value
        elif key == 'ImgList':
            ImgList= value
        elif key == 'DebugPlotsB':
            DebugPlotsB = value
        elif key == 'saveWB':
            saveWB = value 
        elif key == 'HSVrange':
            HSVmin = value[0]
            HSVmax = value[1]
        elif key == "Binfactor1":
            factor1 = value
        elif key == "Binfactor2":
            factor2 = value
        elif key == "Binthreshold" :
            Areath = value
        elif key == "to_gray":
            to_gray = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    DoSelect = False
    DoAlign = False
    DoBackground = False
    DoOpticalFlow = False
    
    if "S" in ToDo:
        DoSelect = True
    if "A" in ToDo:
        DoAlign = True
    if "B" in ToDo:
        DoBackground = True
    if "O" in ToDo :
        DoOpticalFlow = True
    
    if DoSelect:
        OpticalStack, Not2notches, Confined = selectGemmae(P, StackList, size = 300, nimg = 10)
        
        np.savetxt(P+"/OpticalStack.csv",OpticalStack,delimiter =", ",fmt ='% s')
        np.savetxt(P+"/Not2notchesStack.csv",Not2notches,delimiter =", ",fmt ='% s')
        np.savetxt(P+"/ConfinedStack.csv",Confined,delimiter =", ",fmt ='% s')
    
    else :
        OpticalStack = np.loadtxt(P+"/OpticalStack.csv", delimiter =", ", dtype = str)
    
    if DoAlign:
        
        if not os.path.exists(P + '/GlobalData' + stringName + '_Landmarks.csv'):
            if not os.path.exists(P + '/GlobalData' + stringName + '_Landmarks_tmp.csv'):
                ContourData = pd.read_csv(P + '/ContourData' + stringName + '_AreaFit.csv', index_col = 'Ind')
                GlobalData = pd.read_csv(P + '/GlobalData' + stringName + '_AreaFit.csv', index_col = 'Ind')
                print('\n Loaded AreaFit file.')
            else:            
                ContourData = pd.read_csv(P + '/ContourData' + stringName + '_Landmarks_tmp.csv', index_col = 'Ind')
                GlobalData = pd.read_csv(P + '/GlobalData' + stringName + '_Landmarks_tmp.csv', index_col = 'Ind')
                print('\n Loaded Landmarks_tmp file.')
        else:            
            ContourData = pd.read_csv(P + '/ContourData' + stringName + '_Landmarks.csv', index_col = 'Ind')
            GlobalData = pd.read_csv(P + '/GlobalData' + stringName + '_Landmarks.csv', index_col = 'Ind')
            print('\n Loaded Landmarks file.')

        OpticalStack = [s for s in OpticalStack if (s in np.unique(GlobalData.index.values)) ]
        
        ContourData_LM, GlobalData_LM =getLandmarks(ContourData,GlobalData,OpticalStack,Scale,P,stringName, FirstSlice=False, Overwrite = overwriteRefs)
        
        #getNotchesAttach_initial(P,OpticalStack, overwrite = overwriteRefs)
        if os.path.exists(P + '/GlobalData' + stringName + '_Landmarks_aligned.csv'):
            GlobalData_LM = pd.read_csv(P + '/GlobalData' + stringName + '_Landmarks_aligned.csv', index_col = 'Ind')
            
        RotateTranslate_Automatic(GlobalData_LM, OpticalStack,P, stringName, factor = factorA, DebugPlots = DebugPlotsA, ImgList = ImgList)
        
    if not DoAlign : 
        GlobalData = pd.read_csv(P + '/GlobalData' + stringName + '_Landmarks_tmp.csv', index_col = 'Ind')
        OpticalStack = [s for s in OpticalStack if (s in np.unique(GlobalData.index.values)) ]
    if DoBackground:
        RemoveBackgroundStack(OpticalStack, P, Scale, DebugPlots = DebugPlotsB, ImgList = ImgList,saveWB = saveWB, HSVrange = [HSVmin, HSVmax], Binfactor1 = factor1, Binfactor2 = factor2, Binthreshold = Areath)
        
    if DoOpticalFlow:
    
        if not os.path.exists(P + '/Flow'):
                os.mkdir(P + '/Flow') 
    
        for s in OpticalStack:
    
            print('Processing ' + s + ' :')  
            
            if not os.path.exists(P + '/Flow/' + s + '_Flow'):
                os.mkdir(P + '/Flow/' + s + '_Flow')
            
            OpticalFlow(P, s, to_gray=to_gray)
    
    return

def MainOpticalFlow(P, StackList, Scale, ToDo, **kwargs):
    
    # keyword arguments to find reference point
    overwriteRefs = False
    
    # keyword arguments for rotate/translate image
    factorA = [0.2]*len(StackList)
    DebugPlotsA = False
    
    
    ImgList = [0, 20, 40]
    
    # keyword argument for remove background
    DebugPlotsB = False                            
    HSVmin = (30, 20, 70)
    HSVmax = (60, 120,220)
    saveWB = False
    factor1 = 2
    factor2 = 30
    Areath = 5*1e3
    
    # keyword arguments for optical flow
    to_gray = True
    
    for key, value in kwargs.items(): 
        if key == 'overwriteRefs':
            overwriteRefs = value
        elif key == 'factorA':
            factorA = value
        elif key == 'DebugPlotsA':
            DebugPlotsA = value
        elif key == 'ImgList':
            ImgList= value
        elif key == 'DebugPlotsB':
            DebugPlotsB = value
        elif key == 'saveWB':
            saveWB = value 
        elif key == 'HSVrange':
            HSVmin = value[0]
            HSVmax = value[1]
        elif key == "Binfactor1":
            factor1 = value
        elif key == "Binfactor2":
            factor2 = value
        elif key == "Binthreshold" :
            Areath = value
        elif key == "to_gray":
            to_gray = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    DoSelect = False
    DoAlign = False
    DoBackground = False
    DoOpticalFlow = False
    
    if "S" in ToDo:
        DoSelect = True
    if "A" in ToDo:
        DoAlign = True
    if "B" in ToDo:
        DoBackground = True
    if "O" in ToDo :
        DoOpticalFlow = True
    
    if DoSelect:
        OpticalStack, Not2notches, Confined = selectGemmae(P, StackList, size = 300, nimg = 10)
        
        np.savetxt(P+"OpticalStack.csv",OpticalStack,delimiter =", ",fmt ='% s')
        np.savetxt(P+"Not2notchesStack.csv",Not2notches,delimiter =", ",fmt ='% s')
        np.savetxt(P+"ConfinedStack.csv",Confined,delimiter =", ",fmt ='% s')
    
    else :
        OpticalStack = np.loadtxt(P+"OpticalStack.csv", delimiter =", ", dtype = str)
    
    if DoAlign:
        getNotchesAttach_initial(P,OpticalStack, overwrite = overwriteRefs)
        RotateTranslate(OpticalStack,P, factor = factorA, DebugPlots = DebugPlotsA, ImgList = ImgList)
        
    if DoBackground:
        RemoveBackgroundStack(OpticalStack, P, Scale, DebugPlots = DebugPlotsB, ImgList = ImgList,saveWB = saveWB, HSVrange = [HSVmin, HSVmax], Binfactor1 = factor1, Binfactor2 = factor2, Binthreshold = Areath)
        
    if DoOpticalFlow:
    
        if not os.path.exists(P + 'Flow'):
                os.mkdir(P + 'Flow') 
    
        for s in OpticalStack:
    
            print('Processing ' + s + ' :')  
            
            if not os.path.exists(P + '/Flow/' + s + '_Flow'):
                os.mkdir(P + '/Flow/' + s + '_Flow')
            
            OpticalFlow(P, s, to_gray=to_gray)
    
    return
        
    
def OpticalFlow(P, s, to_gray=True):
    
    ProcessedPath = P + '/Processed/' + s + '_NoBackground/'
    FlowPath = P + '/Flow/' + s + '_Flow/'

    nimg = len([entry for entry in os.listdir(ProcessedPath) if os.path.isfile(os.path.join(ProcessedPath, entry))]) # number of images
    
    prev_frame = io.imread(ProcessedPath + '/' + str(0) + '.tif')
    
    if to_gray :
        # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
        prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    else :
        prev_gray = prev_frame
    
    
        
    for i in range(1,nimg):
        print('Measuring optical flow ' + str(i+1) + '/' + str(nimg).ljust(10), flush=True, end = '\r')
            
        prev_frame = io.imread(ProcessedPath + '/' + str(i-1) + '.tif')
        frame = io.imread(ProcessedPath + '/' + str(i) + '.tif')
        
        if to_gray :
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else :
            gray = frame
            
        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                        None,
                                        0.5, 3, 15, 3, 5, 1.2, 1.1) # look more how to set the parameters
        
        io.imsave(FlowPath + str(i) + '_U.tif', flow[..., 0], plugin='tifffile')
        io.imsave(FlowPath + str(i) + '_V.tif', flow[..., 1], plugin='tifffile')
        
        prev_gray = gray
    
    return     

def OpticalFlow_Bckg(P, s, to_gray=True):
    
    ProcessedPath = P + '/Aligned/' + s + '_aligned/'
    FlowPath = P + '/Flow/' + s + '_Flow/'

    nimg = len([entry for entry in os.listdir(ProcessedPath) if os.path.isfile(os.path.join(ProcessedPath, entry))]) # number of images
    
    prev_frame = io.imread(ProcessedPath + '/' + str(0) + '.tif')
    
    if to_gray :
        # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
        prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    else :
        prev_gray = prev_frame
    
    
        
    for i in range(1,nimg):
        print('Measuring optical flow ' + str(i+1) + '/' + str(nimg).ljust(10), flush=True, end = '\r')
            
        prev_frame = io.imread(ProcessedPath + '/' + str(i-1) + '.tif')
        frame = io.imread(ProcessedPath + '/' + str(i) + '.tif')
        
        if to_gray :
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else :
            gray = frame
            
        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray,None,
                                           pyr_scale = 0.5,
                                           levels = 3,
                                           winsize = 15,
                                           iterations = 3,
                                           poly_n = 5,
                                           poly_sigma = 1.1,
                                           flags = cv.OPTFLOW_FARNEBACK_GAUSSIAN) # look more how to set the parameters
        
        io.imsave(FlowPath + str(i) + '_U.tif', flow[..., 0], plugin='tifffile')
        io.imsave(FlowPath + str(i) + '_V.tif', flow[..., 1], plugin='tifffile')
        
        prev_gray = gray
    
    return     


#%%

def drawFlowIndi(P, Gemma, FPH, Delay, TimeFrame, nvec = 20, scale = 4):
    Pflow = P + '/Flow/' + str(Gemma) + '_FlowNoBackground/'
        
    down, up = TimeFrame
    FirstImg = int((down - Delay)//(60/FPH))
    if FirstImg < 1 :
        FirstImg = 1
    LastImg = int((up - Delay)//(60/FPH))
        
        
    U_list = []
    V_list = []
    
    
    for i in range(FirstImg, LastImg+1):
        
        U_list.append(io.imread(Pflow+str(i)+"_U.tif"))
        V_list.append(io.imread(Pflow+str(i)+"_V.tif"))

    U = np.nansum(U_list, axis = 0)
    V = np.nansum(V_list, axis = 0)
    Norm = np.sqrt(np.nansum([np.multiply(U,U),np.multiply(V,V)], axis = 0))
        
    nl, nc = Norm.shape
    step = max(nl//nvec, nc//nvec)

    u_ = U[::step, ::step]
    v_ = V[::step, ::step]
    
    
    
    
    fig,ax = plt.subplots(dpi = 250,facecolor='white')
    

    ax.imshow(Norm, cmap = 'Blues')
    y, x = np.mgrid[:nl:step, :nc:step]
    ax.quiver(x, y, u_, v_, Norm[::step, ::step], units='dots',
               angles='xy', scale_units='xy', lw=3, cmap = 'Reds', scale = scale)
    ax.set_title('Optical flow for '+ str(Gemma)+ ' between '+str(down)+ ' and '+ str(up)+ ' min')
    fig.tight_layout()
    plt.show()
    
    return
    
def Gemmae_fourth_opticalFlow(GD, stackList, P, FPH, Delay, TimeFrame):
    
    # initiate a dataframe
    df = pd.DataFrame()

    for s in stackList :
        Pflow = P + '/Flow/' + str(s) + '_FlowNoBackground/'
        
        # Define time frame of interest
        down, up = TimeFrame
        FirstImg = int((down - Delay)//(60/FPH))
        if FirstImg < 1 :
            FirstImg = 1
        LastImg = int((up - Delay)//(60/FPH))
        
        U_list = []
        V_list = []
        
        # Calculate overall displacement over the time frame
        for i in range(FirstImg, LastImg+1):
        
            U_list.append(io.imread(Pflow+str(i)+"_U.tif"))
            V_list.append(io.imread(Pflow+str(i)+"_V.tif"))

        U = np.nansum(U_list, axis = 0)
        V = np.nansum(V_list, axis = 0)
        Norm = np.sqrt(np.nansum([np.multiply(U,U),np.multiply(V,V)], axis = 0))
        
        ### Cut in fourth
        # Find reference points
        Yn1Al = GD.loc[(GD.index == s) & (GD['Img'] == i),'Yn1Al']
        Xn1Al = float(GD.loc[(GD.index == s) & (GD['Img'] == i),'Xn1Al'])
        Xn2Al = float(GD.loc[(GD.index == s) & (GD['Img'] == i),'Xn2Al'])
        # set X and Y thresholds
        thY = int(Yn1Al)
        thX = int(min(Xn1Al, Xn2Al)+ abs(Xn1Al-Xn2Al)/2)
        
        U1 = U[0:thY,0:thX]
        U1 = U1[U1!=0]
        U2 = U[thY:,0:thX]
        U2 = U2[U2!=0]
        U3 = U[0:thY,thX:]
        U3 = U3[U3!=0]
        U4 = U[thY:,thX:]
        U4 = U4[U4!=0]
        
        V1 = V[0:thY,0:thX]
        V1 = V1[V1!=0]
        V2 = V[thY:,0:thX]
        V2 = V2[V2!=0]
        V3 = V[0:thY,thX:]
        V3 = V3[V3!=0]
        V4 = V[thY:,thX:]
        V4 = V4[V4!=0]
        
        Norm1 = Norm[0:thY,0:thX]
        Norm1 = Norm1[Norm1!=0]
        Norm2 = Norm[thY:,0:thX]
        Norm2 = Norm2[Norm2!=0]
        Norm3 = Norm[0:thY,thX:]
        Norm3 = Norm3[Norm3!=0]
        Norm4 = Norm[thY:,thX:]
        Norm4 = Norm4[Norm4!=0]
        
        data = {'TimeInit': TimeFrame[0],
                'TimeEnd': TimeFrame[1],
                'median_U1': np.median(U1)/np.shape(U1)[0],
                'median_U2': np.median(U2)/np.shape(U2)[0],
                'median_U3': np.median(U3)/np.shape(U2)[0],
                'median_U4': np.median(U4)/np.shape(U2)[0],
                'std_U1': np.std(U1)/np.shape(U1)[0],
                'std_U2': np.std(U2)/np.shape(U2)[0],
                'std_U3': np.std(U3)/np.shape(U2)[0],
                'std_U4': np.std(U4)/np.shape(U2)[0],
                'median_V1': np.median(V1)/np.shape(V1)[0],
                'median_V2': np.median(V2)/np.shape(V2)[0],
                'median_V3': np.median(V3)/np.shape(V2)[0],
                'median_V4': np.median(V4)/np.shape(V2)[0],
                'std_V1': np.std(V1)/np.shape(V1)[0],
                'std_V2': np.std(V2)/np.shape(V2)[0],
                'std_V3': np.std(V3)/np.shape(V2)[0],
                'std_V4': np.std(V4)/np.shape(V2)[0],
                'median_Norm1': np.median(Norm1)/np.shape(Norm1)[0],
                'median_Norm2': np.median(Norm2)/np.shape(Norm2)[0],
                'median_Norm3': np.median(Norm3)/np.shape(Norm2)[0],
                'median_Norm4': np.median(Norm4)/np.shape(Norm2)[0],
                'std_Norm1': np.std(Norm1)/np.shape(Norm1)[0],
                'std_Norm2': np.std(Norm2)/np.shape(Norm2)[0],
                'std_Norm3': np.std(Norm3)/np.shape(Norm2)[0],
                'std_Norm4': np.std(Norm4)/np.shape(Norm2)[0],
                }            

        df = df.append(pd.DataFrame(data=data,index = [s]))
        
        
    fig = plt.figure(figsize=(10, 10))
    outer = mpl.gridspec.GridSpec(1, 2, figure = fig)

    theta1, theta2, theta3, theta4 = 0, 90, 180, 270
    radius = 2
    center = (0, 0)
    
    # Median
    #inner11 = mpl.gridspec.GridSpecFromSubplotSpec(1, 1,
     #               subplot_spec=outer[0], wspace=0.1, hspace=0.1)
    
    
    inner1 = mpl.gridspec.GridSpecFromSubplotSpec(3, 1,
                    subplot_spec=outer[0])
    
    inner10 = fig.add_subplot(inner1[0])
    inner11 = fig.add_subplot(inner1[1])
    inner12 = fig.add_subplot(inner1[2])
    
    BlueMap = mpl.cm.get_cmap('Blues')
    
    w1 = mpl.patches.Wedge(center, radius, theta2, theta3, fc=BlueMap(df.median()['median_Norm1']/0.002), edgecolor='black')
    w2 = mpl.patches.Wedge(center, radius, theta3, theta4, fc=BlueMap(df.median()['median_Norm2']/0.002), edgecolor='black')
    w3 = mpl.patches.Wedge(center, radius, theta1, theta2, fc=BlueMap(df.median()['median_Norm3']/0.002), edgecolor='black')
    w4 = mpl.patches.Wedge(center, radius, theta4, theta1, fc=BlueMap(df.median()['median_Norm4']/0.002), edgecolor='black')
    
    for wedge in [w1, w2, w3, w4]:
       inner10.add_artist(wedge)
    
    x = [-1, -1, 1, 1]
    y = [1,-1,1,-1]
    
    u_ = [df.mean()['median_U1'],-df.mean()['median_U3'],-df.mean()['median_U2'],df.mean()['median_U4']]
    
    
    v_ = [-df.mean()['median_V1'],df.mean()['median_V3'],df.mean()['median_V2'],-df.mean()['median_V4']]
    
    Norm = [df.mean()['median_Norm1'],df.mean()['median_Norm3'],df.mean()['median_Norm2'],df.mean()['median_Norm4']]
    
    scale = 0.0005
    
    vect = inner10.quiver(x,y,u_, v_, Norm, units='dots',
                   angles='xy', scale_units='xy', lw=5, cmap = 'Reds', scale = scale)
    
    
    inner10.text(-1.8,0.2, str(round(df.median()['median_Norm1'], 6)), fontsize = 7)
    inner10.text(-1.8,-0.4, str(round(df.median()['median_Norm2'], 6)), fontsize = 7)
    inner10.text(0.2,0.2, str(round(df.median()['median_Norm3'], 6)), fontsize = 7)
    inner10.text(0.2,-0.4, str(round(df.median()['median_Norm4'], 6)), fontsize = 7)
    
    fig.colorbar(vect, ax = inner10)
    inner10.axis('equal')
    inner10.set_xbound(lower = -4, upper = 4)
    inner10.set_ybound(-4, 4)
    inner10.set_adjustable('box')
    inner10.set_title('Median of median Norm and direction')
    
    df['Q1vsQ3'] = abs(df['median_Norm1']-df['median_Norm3'])/np.mean([df['median_Norm1'],df['median_Norm3']])
    df['Q2vsQ4'] = abs(df['median_Norm2']-df['median_Norm4'])/np.mean([df['median_Norm2'],df['median_Norm4']])
    
    inner11.hist(df['Q1vsQ3'].values)
    inner11.set_title('Diff. Q1 vs Q3')
    
    inner12.hist(df['Q2vsQ4'].values)
    inner12.set_title('Diff. Q2 vs Q4')
    
    
    # Variance
    inner2 = mpl.gridspec.GridSpecFromSubplotSpec(3, 1,
                    subplot_spec=outer[1])
    
    
    BlueMap = mpl.cm.get_cmap('Blues')
    
    center = (0, 0)

    inner20 = fig.add_subplot(inner2[0])
    inner21 = fig.add_subplot(inner2[1])
    inner22 = fig.add_subplot(inner2[2])
    
    w1 = mpl.patches.Wedge(center, radius, theta2, theta3, fc=BlueMap(df.median()['std_Norm1']/abs(df.median()['median_Norm1'])*0.5), edgecolor='black')
    w2 = mpl.patches.Wedge(center, radius, theta3, theta4, fc=BlueMap(df.median()['std_Norm2']/abs(df.median()['median_Norm2'])*0.5), edgecolor='black')
    w3 = mpl.patches.Wedge(center, radius, theta1, theta2, fc=BlueMap(df.median()['std_Norm3']/abs(df.median()['median_Norm3'])*0.5), edgecolor='black')
    w4 = mpl.patches.Wedge(center, radius, theta4, theta1, fc=BlueMap(df.median()['std_Norm4']/abs(df.median()['median_Norm4'])*0.5), edgecolor='black')
    
    for wedge in [w1, w2, w3, w4]:
        inner20.add_artist(wedge)
    
    inner20.text(-1.8,0.2, str(round(df.median()['std_Norm1']/abs(df.median()['median_Norm1']), 6)), fontsize = 7)
    inner20.text(-1.8,-0.4, str(round(df.median()['std_Norm2']/abs(df.median()['median_Norm2']), 6)), fontsize = 7)
    inner20.text(0.2,0.2, str(round(df.median()['std_Norm3']/abs(df.median()['median_Norm3']), 6)), fontsize = 7)
    inner20.text(0.2,-0.4, str(round(df.median()['std_Norm4']/abs(df.median()['median_Norm4']), 6)), fontsize = 7)
    
    inner20.axis('equal')
    inner20.set_xbound(-4, 4)
    inner20.set_ybound(-4, 4)
    inner20.set_title('Median Norm std')
    inner20.set_adjustable('box')
    

    
    w1 = mpl.patches.Wedge(center, radius, theta2, theta3, fc=BlueMap(df.median()['std_U1']/abs(df.median()['median_U1'])*0.5), edgecolor='black')
    w2 = mpl.patches.Wedge(center, radius, theta3, theta4, fc=BlueMap(df.median()['std_U2']/abs(df.median()['median_U2'])*0.5), edgecolor='black')
    w3 = mpl.patches.Wedge(center, radius, theta1, theta2, fc=BlueMap(df.median()['std_U3']/abs(df.median()['median_U3'])*0.5), edgecolor='black')
    w4 = mpl.patches.Wedge(center, radius, theta4, theta1, fc=BlueMap(df.median()['std_U4']/abs(df.median()['median_U4'])*0.5), edgecolor='black')
     
    for wedge in [w1, w2, w3, w4]:
       inner21.add_artist(wedge)
    
    inner21.text(-1.8,0.2, str(round(df.median()['std_U1']/abs(df.median()['median_U1']), 6)), fontsize = 7)
    inner21.text(-1.8,-0.4, str(round(df.median()['std_U2']/abs(df.median()['median_U2']), 6)), fontsize = 7)
    inner21.text(0.2,0.2, str(round(df.median()['std_U3']/abs(df.median()['median_U3']), 6)), fontsize = 7)
    inner21.text(0.2,-0.4, str(round(df.median()['std_U4']/abs(df.median()['median_U4']), 6)), fontsize = 7)
       
    inner21.axis('equal')
    inner21.set_xbound(-4, 4)
    inner21.set_ybound(-4, 4)
    inner21.set_title('Median U std')
    inner21.set_adjustable('box')
    

       
    w1 = mpl.patches.Wedge(center, radius, theta2, theta3, fc=BlueMap(df.median()['std_V1']/abs(df.median()['median_V1'])*0.5), edgecolor='black')
    w2 = mpl.patches.Wedge(center, radius, theta3, theta4, fc=BlueMap(df.median()['std_V2']/abs(df.median()['median_V2'])*0.5), edgecolor='black')
    w3 = mpl.patches.Wedge(center, radius, theta1, theta2, fc=BlueMap(df.median()['std_V3']/abs(df.median()['median_V3'])*0.5), edgecolor='black')
    w4 = mpl.patches.Wedge(center, radius, theta4, theta1, fc=BlueMap(df.median()['std_V4']/abs(df.median()['median_V4'])*0.5), edgecolor='black')
     
    for wedge in [w1, w2, w3, w4]:
       inner22.add_artist(wedge)
    
    
    inner22.text(-1.8,0.2, str(round(df.median()['std_V1']/abs(df.median()['median_V1']), 6)), fontsize = 7)
    inner22.text(-1.8,-0.4, str(round(df.median()['std_V2']/abs(df.median()['median_V2']), 6)), fontsize = 7)
    inner22.text(0.2,0.2, str(round(df.median()['std_V3']/abs(df.median()['median_V3']), 6)), fontsize = 7)
    inner22.text(0.2,-0.4, str(round(df.median()['std_V4']/abs(df.median()['median_V4']), 6)), fontsize = 7)
    
    inner22.axis('equal')
    inner22.set_xbound(-4, 4)
    inner22.set_ybound(-4, 4)
    inner22.set_title('Median V std')
    inner22.set_adjustable('box')
    
    # try to do median angle/direction
    
    

    plt.show()
    
    return df
        

#%%
"""
def compareFlow(Ps, Stacklists, FPHs, Delays, TimeFrames, nvec = 20, factor = 1.2, overwrite = False):
    # to do = calculate overall flow in the TimeFrames for each condition = 
    # plot the median, the variance for each condition and the difference between the two conditions
    
    XAs_ref = []
    Yn1s_ref = []
    median_Us = []
    median_Vs = []
    median_Norms = []

    indiplots = False
    
    # calculate the overallflow for each gemmae > keep median and variance
    for P, Stacklist, FPH, Delay, TimeFrame in zip(Ps, Stacklists, FPHs, Delays, TimeFrames):
        Pflow = P + '/Flow/'
        
        down, up = TimeFrame
        FirstImg = (down - Delay)//(60/FPH)
        LasttImg = (up - Delay)//(60/FPH)
        
        
        Norm_list = []
        U_list = []
        V_list = []
        max_dim = (0,0)
        
        for s in Stacklist:
            Im = io.imread(Pflow+str(s)+"_Flow/"+str(0)+"_U.tif")
            max_dim = np.max([max_dim, np.shape(Im)], axis = 0) # il faut prendre les distance depuis YA
            
        
        RefPts = pd.read_csv(P + '/clickedpoints_alltimes.csv', index_col = 'Ind')

        
        # trouver les dimensions max ? > trouver les dimensions adaptée et les mettre dans la taille d'arrivée de la matrix de translation
        # par ex = les plus grandes dimensions * 1.5...
        
        for s in Stacklist:
        
            U_list = []
            V_list = []
            nimg = len([entry for entry in os.listdir(ProcessedPath) if os.path.isfile(os.path.join(ProcessedPath, entry))])/2 # number of images

            for i in range(FirstImg-1, LastImg+1):
                U_list.append(io.imread(Pflow+str(s)+"_Flow/"+str(i)+"_U.tif"))
                V_list.append(io.imread(Pflow+str(s)+"_Flow/"+str(i)+"_V.tif"))

            U = np.sum(U_list, axis = 0)
            V = np.sum(V_list, axis = 0)
            Norm = np.sqrt(np.sum([np.multiply(U,U),np.multiply(V,V)], axis = 0))
            
            
            XA = RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == 0), 'Xattach'].value
            Yn1 = RefPts.loc[(RefPts.index == s) & (RefPts['Img'] == 0), 'Ynotch1'].value
            
            if s == 0:
                XA_ref = XA
                Yn1_ref = Yn1
                shifted_U = U
                shifted_V = V
                shifted_Norm = Norm
                
                XAs_ref.append(XA_ref)
                Yn1s_ref.append(Yn1_ref)
                
            else :
                M = np.float32([[1,0,XA-XA_ref], [0,1,Yn1-Yn1_ref]])
                shifted_U = cv.warpAffine(U, M, (U.shape[1],U.shape[0]))
                shifted_V = cv.warpAffine(V, M, (V.shape[1],V.shape[0]))
                shifted_Norm = cv.warpAffine(Norm, M, (Norm.shape[1],Norm.shape[0]))
            
            
            Norm_list.append(add_border(shifted_Norm, max_dim*factor))
            U_list.append(add_border(shifted_U, max_dim*factor))
            V_list.append(add_border(shifted_V, max_dim*factor))
            
            if indiplots:
                
                nl, nc = Norm_mean.shape
                step = max(nl//nvec, nc//nvec)

                u_ = U[::step, ::step]
                v_ = V[::step, ::step]

                plt.imshow(Norm_mean, cmap = 'Blues')
                y, x = np.mgrid[:nl:step, :nc:step]
                plt.quiver(x, y, u_, v_, Norm_mean[::step, ::step], units='dots',
                           angles='xy', scale_units='xy', lw=3, cmap = 'Reds', scale = 4)
                 
        
        u_median = np.median(U_list, axis = 0)
        v_median = np.median(V_list, axis = 0)
        norm_median = np.median(Norm_list, axis = 0)
        
        median_Us.append(u_median)
        median_Vs.append(v_median)
        median_Norms.append(norm_median)
        
        # faire option pour vérifier que le fichier n'existe pas déjà ?
        io.imsave(PFlow + "median_U_from_"+ str(down) + 'min_to_' + str(up) + 'min.tif', u_median, plugin='tifffile')
        io.imsave(PFlow + "median_V_from_"+ str(down) + 'min_to_' + str(up) + 'min.tif', v_median, plugin='tifffile')
        io.imsave(PFlow + "median_Norm_from_"+ str(down) + 'min_to_' + str(up) + 'min.tif', norm_median, plugin='tifffile')
        
        if conditionplots:
                
                nl, nc = norm_median.shape
                step = max(nl//nvec, nc//nvec)

                u_ = u_median[::step, ::step]
                v_ = v_median[::step, ::step]

                plt.imshow(norm_median, cmap = 'Blues')
                y, x = np.mgrid[:nl:step, :nc:step]
                plt.quiver(x, y, v_median, v_median, norm_median[::step, ::step], units='dots',
                           angles='xy', scale_units='xy', lw=3, cmap = 'Reds', scale = 4)
                
    if len(Ps) == 2:
        
        # voir s'il ne faut pas normaliser la taille initiale ?
        max_dim = np.max(np.shape(median_Us[0]),np.shape(median_Us[0]), axis = 0)
        
        median_Us = [add_border(U, max_dim*factor) for U in median_Us]
        median_Vs = [add_border(V, max_dim*factor) for V in median_Vs]
        median_Norms = [add_border(Norm, max_dim*factor) for Norm in median_Norms]
        
        # enregistrer les plots puis comparer entre conditions
        
        # to compare the two conditions = have to use the XA and YA of the first image
    return
            
"""         


        

    


