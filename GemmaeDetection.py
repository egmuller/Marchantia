# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:34:34 2022

@author: laplaud
"""


# Imports

#  Number and vectors and data
import numpy as np
import pandas as pd

# Plotting
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from cycler import cycler

# Signal and geometry analysis
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d

# Image analysis
import cv2 as cv

from skimage import io
from skimage.morphology import binary_opening, binary_closing, remove_small_holes
from skimage.measure import regionprops

import tifffile as tifff


# other stuff
from IPython import get_ipython

import os

#%% Creation of circular mask for morphology operations

# Utility function that creates a cirular binary mask on a rectangular image

# h : height of binary image, w : width of binary image, center : center of mask,
# radius : radius of mask


def create_circular_mask(h, w, center=None, radius=None):


    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


#%% Binarization of RGB image

# Transform the RGB image into a HSV image, and then thresholds it using the 
# two boundary triplets HSVmin and HSVmax. Binary morphological operations are 
# then applied to clean up the shape.

# Img : RGB image to binarize, Scale : in µm/px, HSVmin/max : triples of 
# thershold for HSV image. 

# Kwargs : 'debug' (True/False) for displaying debug plots


def Binarize(Img, Scale, HSVmin, HSVmax, **kwargs):

    #init and read kwargs
    DebugPlots = False
    
    for key, value in kwargs.items(): 
        if key == 'debug':
            DebugPlots = value
            

    
    hsv = cv.cvtColor(Img, cv.COLOR_RGB2HSV)

    ## mask 
    mask = cv.inRange(hsv, HSVmin, HSVmax)

    ## slice 
    BWimg = mask>0
    
    size = np.round(2*Scale) # 5µm in pixels
    
    selem = create_circular_mask(size,size) # create circular element for closing

    DilBWimg = binary_closing(BWimg,selem) # image closing
    
    FilledBWimg = remove_small_holes(DilBWimg, area_threshold=5*1e3) # fills dark regions
    
    Size = np.round(30*Scale) # 30µm in pixels
    
    Selem = create_circular_mask(Size,Size) # create circular element for opening

    FinalImg = binary_opening(FilledBWimg,Selem) # image opening
    
    if DebugPlots:        
       
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
    
    return(FinalImg)

#%% Binarize stacks function

# Applies binarization to a list of image stacks. First 
# performs an harmonisation of intensity across all images in one stack, then 
# a white balance based on user input, then binarization of all images in the stack.
# Saving is done in a specific folder placed in the original folder containing 
# the stacks to binarze, white balance data are saved in a .csv file that can 
# be reloaded for code rerun without user input. 

# StackList : list of stacks names to analyse, P : path to data, Scale : in µm/px

# Kwargs : 'debug' (True/False) for plotting debug plots, 'debuglist' (list of integers) to choose the images to analyse for debug plots
# 'HSVrange' (list of two 1x3 tuples) binarization min and max threshold values for the three channels of HSV image 

def BinarizeStack(StackList, P, Scale, **kwargs):

    #init and read kwargs
    DebugPlots = False                            
    HSVmin = (30, 20, 70)
    HSVmax = (60, 120,220)
    ImgList = [0, 20, 40]
    
    for key, value in kwargs.items(): 
        if key == 'debug':
            DebugPlots = value 
        elif key == 'debuglist':
            ImgList = value 
        elif key == 'HSVrange':
            HSVmin = value[0]
            HSVmax = value[1]
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
     

    # Savefolders for WB and Binary stacks
    if not os.path.exists(P + '\\WhiteBalanced'):
        os.mkdir(P + '\\WhiteBalanced') # create global folder 

    if not os.path.exists(P + '\\Processed'):
        os.mkdir(P + '\\Processed') # create global folder 
            
            
    if os.path.exists(P + '\\WhiteBalanceData.csv'):
        WBsaving = pd.read_csv(P + '\\WhiteBalanceData.csv', index_col = 'Ind')
    else:                
        WBsaving = pd.DataFrame(data=None,columns=['lum','whiteR','whiteG','whiteB']) 
        
    
    
    
    for s in StackList:
        
        if not os.path.exists(P + '\\Processed\\' + s + '_Binarized'):
            # shutil.rmtree(P + '\\' + s + '_Binarized') # remove folder and contents
            os.mkdir(P + '\\Processed\\' + s + '_Binarized') # create binarize image folder
            isBin = False
        else:
            isBin = True
            
        print('Processing ' + s + ' :')    

        RGBstack = io.imread(P + '\\' + s + '.tif') # get the tiff stack
        Bckp_RGBstack = io.imread(P + '\\' + s + '.tif') # get the tiff stack for comparison
        
        if DebugPlots:            
            imglist = ImgList
        else:
            imglist = range(len(RGBstack))  
            #imglist = range(49)  
            
        #### Adjusting brightness
        # getting reference value
        HSVref = cv.cvtColor(RGBstack[0,:,:,:], cv.COLOR_BGR2HSV)
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

            WBsaving.to_csv(P + '\\WhiteBalanceData.csv',index_label = 'Ind')
            
        else:
            whiteR = WBsaving.loc[s,'whiteR']
            whiteG = WBsaving.loc[s,'whiteG']
            whiteB = WBsaving.loc[s,'whiteB']
            lum = WBsaving.loc[s,'lum']


            
        # adjusting stack white balance
        RGBstack[:,:,:,0] = RGBstack[:,:,:,0] * lum / whiteR
        RGBstack[:,:,:,1] = RGBstack[:,:,:,1] * lum / whiteG
        RGBstack[:,:,:,2] = RGBstack[:,:,:,2] * lum / whiteB

        
        tifff.imsave(P + '\\WhiteBalanced\\' + s + '.tif', RGBstack)
            


        
        
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
                
                BinImg = Binarize(Img,Scale,HSVmin,HSVmax,debug =DebugPlots)
                
                io.imsave(P + '\\Processed\\' + s + '_Binarized\\' + str(i) + '.tif', np.uint8(BinImg*255), plugin='tifffile')
            
            else:
                if os.path.exists(P + '\\Processed\\' + s + '_Binarized\\' + str(i) + '.tif'):
                    BinImg = io.imread(P + '\\Processed\\' + s + '_Binarized\\' + str(i) + '.tif') 
                else:
                    print('Binarization of image ' + str(i+1) + '/' + str(len(RGBstack)).ljust(15), flush=True, end = '\r')
                    
                    BinImg = Binarize(Img,Scale,HSVmin,HSVmax,debug =DebugPlots)
                    io.imsave(P + '\\Processed\\' + s + '_Binarized\\' + str(i) + '.tif', np.uint8(BinImg*255), plugin='tifffile')
            


        print('Binarization done for ' + str(len(RGBstack)) + '/' + str(len(RGBstack)) + ' images'.ljust(20), flush=True)

        
    return


#%% Getting edge and area of gemmae binary mask

# Identify the largest object on binary image, isolates it, and saves it area 
# and contour. Also compute the geometrical center using a distance transform.

# BinImg : binary image on which to perform identification, Scale : in µm/px

def getEdgeAndArea(BinImg,Scale):
    
    # Find largest contour 
    cnts, _ = cv.findContours(BinImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv.contourArea)

    Area = cv.contourArea(cnt)*Scale**2

    # Create image with only largest object
    out = np.zeros(BinImg.shape, np.uint8)
    cv.drawContours(out, [cnt], -1, 255, cv.FILLED)
    BinImg = cv.bitwise_and(BinImg, out)

    # Computing center of propagule using euclidian distance transform
    dist = cv.distanceTransform(BinImg, cv.DIST_L2, 3)
    regions = regionprops(BinImg,dist)
    center = regions[0].weighted_centroid

    # Getting contour 
    Ycnt = cnt[:,0,0]
    Xcnt = cnt[:,0,1] 
    
    ContourCumLength = np.concatenate(([0],np.cumsum(np.sqrt(np.square(np.diff(Xcnt))+np.square(np.diff(Ycnt))))))
    ContourLength = ContourCumLength[-1]
        
    ContourInterp = interp1d(ContourCumLength,[Xcnt,Ycnt], fill_value='extrapolate')            

    npts = 1000

    # create contour with regular points
    deltaS = ContourLength/npts
    ContourRegCumLength = np.linspace(0,npts,npts+1)*deltaS

    RegXcnt,RegYcnt = ContourInterp(ContourRegCumLength)
    
    # Edge coordinate relative to the center
    relativeIndicesX = RegXcnt-center[0]
    relativeIndicesY = RegYcnt-center[1]      
    
    Xlength = relativeIndicesX.max() - relativeIndicesX.min()
    Ylength = relativeIndicesY.max() - relativeIndicesY.min()
    
    return(relativeIndicesX,relativeIndicesY,center,Area,Xlength,Ylength)


#%% Area & contours from stacks

# Analyze stacks and pool contour and area data into a panda dataframe.

# StackList : list of stacks names to analyse, P : path to data, 
# Scale : in µm/px, FPH : Frames per Hours

# Kwargs : 'debug' (True/False) for displaying debug plots

def GetContours(StackList,P, Scale, FPH, **kwargs):
      
    # init and read kwargs    
    DebugPlots = False
    
    for key, value in kwargs.items(): 
        if key == 'debug':
            DebugPlots = value
        else:
            print('Unknown key : ' + key + '. Kwarg ignored.')
    
    # initialize dataframme
    GD = pd.DataFrame(data=None,columns=['Img','Time (min)','Area','Xcenter','Ycenter']) 

    CD = pd.DataFrame(data=None,columns=['Img','Ximg','Yimg']) 

    
    for s in StackList:

        print('Processing ' + s + ' :')   
        
        ProcessedPath = P + '\\Processed\\' + s + '_Binarized\\'

        n = len([entry for entry in os.listdir(ProcessedPath) if os.path.isfile(os.path.join(ProcessedPath, entry))]) # number of images

        jet_colorcycle = [plt.get_cmap('jet')(1. * i/n) for i in range(n)]
        mpl.rcParams['axes.prop_cycle'] = cycler(color=jet_colorcycle)        


        for i in range(n):
            
            print('Measuring contour for image ' + str(i+1) + '/' + str(n).ljust(10), flush=True, end = '\r')
                        
            # Loading binary image
            BinImg = io.imread(ProcessedPath + '\\' + str(i) + '.tif')

            # Computing propagule edge and area from binary image
            SortedX,SortedY,center,Area,Xlength,Ylength = getEdgeAndArea(BinImg,Scale) 
            
            # Storing contour data
            data = {'Ximg':SortedY+center[1],
                        'Yimg':SortedX+center[0],
                        'Img':i*np.ones(len(SortedX))} 

            CD = CD.append(pd.DataFrame(data=data,index = np.repeat(s,len(SortedX))))  # adding to global dataframe
            
            data2 = {'Area':Area/1000000, # In mm²
                     'Xlength':Xlength,
                     'Ylength':Ylength,
                        'Xcenter':center[1],
                        'Ycenter':center[0],
                        'Img':i,
                        'Time (min)':i*60/FPH} 
            
            GD = GD.append(pd.DataFrame(data=data2,index = [s])) 
            
            if ((i == 0)|(i == 1)|(i == 2)) & DebugPlots: #((i == 1)|(i == 21)|(i == 41)) &
                RGBimg = io.imread(P + '\\' + s + '.tif', key = i)

                plt.figure(dpi=250,facecolor='white')
                plt.title(s)
                plt.imshow(RGBimg)
                plt.plot(SortedY+center[1],SortedX+center[0],'c-o',lw = 0.7,ms=1)
                plt.show()
                
        print('Contours saved.'.ljust(35), flush = True)
         
    return(CD,GD)



#%% Finding chip position for every gemmae

def FindChipPos(StackList,Path,Orientation):
    
    FullChip = io.imread(Path + '\\FullChip.tif')
    
    large_image = FullChip[:,:,2]  
    
    L,W = large_image.shape
    
    fig,ax = plt.subplots(dpi=300)
    ax.imshow(FullChip)
    
    positions = np.empty(len(StackList))
    

    
    for s,i in zip(StackList,range(len(StackList))):
        
        small_image = io.imread(Path + '\\' + s + '.tif')[4,:,:,2]
        
        l,w = small_image.shape
        
        method = cv.TM_SQDIFF_NORMED

        result = cv.matchTemplate(small_image, large_image, method)

        _,_,mnLoc,_ = cv.minMaxLoc(result)
    
        MPx,MPy = mnLoc
        
        if Orientation == 'H>':
            pos = int(np.ceil((MPx+w/2)/W*101))
        elif Orientation == 'H<':
            pos = 100-int(np.ceil((MPx+w/2)/W*101))
        elif Orientation == 'Vv':
            pos = int(np.ceil((MPy+l/2)/L*101))
        elif Orientation == 'V^':
            pos = 100-int(np.ceil((MPy+l/2)/L*101))
            
        positions[i] = pos

        ax.text(MPx+w/2, MPy+l/2, s[3:], color = 'r', fontsize = 8)
        
    ax.set_xticks([])
    ax.set_yticks([])

    Datadict = {'Name': StackList, 'Row': positions}
    Data = pd.DataFrame(Datadict)
    Data.to_excel(Path + '\\ChipPositions.xlsx',index=False)
    
    fig.savefig(Path + '\\FullChipTagged.tif')
    plt.close()
    
    return(positions)
    
