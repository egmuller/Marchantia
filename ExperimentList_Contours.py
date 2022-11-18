# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:26:22 2022

@author: laplaud
"""

Scale220301 = 1.94 # Spatial scale (µm/px) for 16X 01-03-2022

Scale220531 = 1.94 # Spatial scale (µm/px) for 16X 31-05-2022
Scale220531_Osmo = 1.94 # Spatial scale (µm/px) for 16X 31-05-2022

Scale220727 = 1.94 # Spatial scale (µm/px) for 16X 27-07-2022


Scale220801 = 1.94 # Spatial scale (µm/px) for 16X 01-08-2022
FPH220801 = 2 # Frames per hour
Delay220801 = 45 # Delay between imbibition and experiment start

Scale220801_Osmo = 1.94 # Spatial scale (µm/px) for 16X 01-08-2022
FPH220801_Osmo = 120 # Frames per hour

Scale220808 = 1.94 # Spatial scale (µm/px) for 16X 01-08-2022
FPH220808 = 2 # Frames per hour
Delay220808 = 45 # Delay between imbibition and experiment start

Scale220808_Osmo = 1.94 # Spatial scale (µm/px) for 16X 01-08-2022
FPH220808_Osmo = 120 # Frames per hour

Scale220928 = 1.94 # Spatial scale (µm/px) for 16X 01-08-2022
FPH220928 = 2 # Frames per hour
Delay220928 = 54 # Delay between imbibition and experiment start

Scale220928_Osmo = 1.94 # Spatial scale (µm/px) for 16X 01-08-2022
FPH220928_Osmo = 120 # Frames per hour

Scale221004_Osmo = 1.94 # Spatial scale (µm/px) for 16X 01-08-2022
FPH221004_Osmo = 120 # Frames per hour


# Paths to data

P220301_Ct = r'D:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220301_DV6_Manitol100mM+OC\TimeEvolution\Ctrl'
P220301_M100 = r'D:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220301_DV6_Manitol100mM+OC\TimeEvolution\Manitol100mM'

P220531_Ct1 = r'D:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220531_DV7_Ctrls+OC\TimeEvolution\Ctrl1'
P220531_Ct2 = r'D:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220531_DV7_Ctrls+OC\TimeEvolution\Ctrl2'
P220531_Ct1_Osmo = r'D:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220531_DV7_Ctrls+OC\OC\Ctrl1'
P220531_Ct2_Osmo = r'D:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220531_DV7_Ctrls+OC\OC\Ctrl2'

P220727_Hyper = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220727_DV7_Ctrls+OC_hyperhypo\TimeEvolution\Hyper'
P220727_Hypo = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220727_DV7_Ctrls+OC_hyperhypo\TimeEvolution\Hypo'

# P220801_Ct1 = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220801_DV7_Ctrls+OC\TimeEvolution\Chamber1'
# P220801_Ct2 = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220801_DV7_Ctrls+OC\TimeEvolution\Chamber2'
# P220801_Ct1_Osmo = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220801_DV7_Ctrls+OC\OC\Chamber1'
# P220801_Ct2_Osmo = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220801_DV7_Ctrls+OC\OC\Chamber2'
# P220808_Ct = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220808_DV7_100mM+OC\TimeEvolution\Ctrl'
# P220808_100mM = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220808_DV7_100mM+OC\TimeEvolution\100mM'
# P220808_Ct_Osmo = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220808_DV7_100mM+OC\OC\Ctrl'
# P220808_100mM_Osmo = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220808_DV7_100mM+OC\OC\100mM'
# P220928_Ct = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220928_DV7_100mM+OC\TimeEvolution\Ctrl'
# P220928_100mM = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220928_DV7_100mM+OC\TimeEvolution\100mM'
# P220928_Ct_Osmo = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220928_DV7_100mM+OC\OC\Ctrl'
# P220928_100mM_Osmo = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\220928_DV7_100mM+OC\OC\100mM'
# P221004_Ct_Osmo = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\221004_DV7_OC_Pectinase10U\Ctrl'
# P221004_Pase10U_Osmo = r'd:\Users\laplaud\Desktop\PostDoc\Data\Microflu\221004_DV7_OC_Pectinase10U\Pase10U'
# PFig = r'D:\Users\laplaud\Desktop\PostDoc\Data\Microflu\Figures'



ExcludeList220301_Ct = []    

ExcludeList220301_M100 = ['PPG24','PPG51']
  

ExcludeList220531_Ct1 = ['PPG19','PPG47','PPG50']    

ExcludeList220531_Ct2 = []
    
ExcludeList220531_Ct1_Osmo =  []

ExcludeList220531_Ct2_Osmo = []


ExcludeList220727_Hyper = ['PPG45','PPG43','PPG48','PPG49','PPG18','PPG22','PPG29']


ExcludeList220727_Hypo = ['PPG7','PPG17','PPG23','PPG31','PPG34','PPG36']


# ExcludeList220801_Ct1 = ['PPG1','PPG2','PPG3','PPG4','PPG5','PPG6','PPG7','PPG8','PPG9','PPG10','PPG11','PPG12','PPG13','PPG14','PPG15','PPG16',
#                         'PPG17','PPG18','PPG19','PPG20','PPG21','PPG22','PPG23','PPG24','PPG25','PPG26','PPG27','PPG28','PPG29','PPG30','PPG31',
#                         'PPG32','PPG33','PPG34','PPG35','PPG36','PPG37','PPG38','PPG39','PPG41','PPG42','PPG43','PPG44','PPG45','PPG46',
#                         'PPG47','PPG49','PPG50','PPG51','PPG52']

    
# ExcludeList220801_Ct2 = ['PPG1','PPG2','PPG3','PPG4','PPG5','PPG6','PPG7','PPG8','PPG9','PPG10','PPG11','PPG12','PPG13','PPG14','PPG15','PPG16',
#                         'PPG17','PPG18','PPG19','PPG20','PPG21','PPG22','PPG23','PPG24','PPG25','PPG26','PPG27','PPG28','PPG29','PPG30','PPG31',
#                                                 'PPG32','PPG33','PPG34','PPG35','PPG36','PPG37','PPG38','PPG39','PPG40']

# ExcludeList220801_Ct2_Osmo = ['PPG1','PPG2','PPG3','PPG4','PPG5','PPG6','PPG8','PPG9','PPG10','PPG11','PPG12','PPG13','PPG14','PPG15','PPG16',
#                         'PPG17','PPG18','PPG19','PPG20','PPG21','PPG22','PPG23','PPG26','PPG27','PPG28','PPG29','PPG30',
#                                                 'PPG32','PPG33','PPG34','PPG35','PPG36','PPG37','PPG38','PPG40']


# ExcludeList220801_Ct1_Osmo = ['PPG1','PPG2','PPG3','PPG4','PPG5','PPG6','PPG7','PPG8','PPG9','PPG10','PPG11','PPG12','PPG13','PPG14','PPG15','PPG16',
#                         'PPG17','PPG18','PPG19','PPG20','PPG21','PPG22','PPG23','PPG26','PPG27','PPG28','PPG29','PPG30','PPG31',
#                         'PPG32','PPG33','PPG34','PPG35','PPG36','PPG37','PPG39','PPG41','PPG42','PPG43','PPG44','PPG45','PPG46',
#                         'PPG47','PPG49','PPG50','PPG51','PPG52','PPGA','PPGB']
    

# ExcludeList220808_Ct = ['PPG1','PPG2','PPG3','PPG4','PPG5','PPG6','PPG7','PPG8','PPG9','PPG10','PPG11','PPG12','PPG13','PPG14','PPG15','PPG16',
#                         'PPG17','PPG18','PPG19','PPG20','PPG21','PPG22','PPG23','PPG24','PPG25','PPG26','PPG27','PPG28','PPG29','PPG30','PPG31',
#                         'PPG32','PPG33','PPG34','PPG35','PPG36','PPG37','PPG38','PPG39','PPG40','PPG41','PPG42','PPG43','PPG44']

    
# ExcludeList220808_100mM = ['PPG1','PPG2','PPG3','PPG4','PPG5','PPG6','PPG7','PPG8','PPG9','PPG10','PPG11','PPG12','PPG13','PPG14','PPG15','PPG16',
#                         'PPG17','PPG18','PPG19','PPG20','PPG21','PPG22','PPG23','PPG24','PPG25','PPG26','PPG27','PPG28','PPG29','PPG30','PPG31',
#                         'PPG32','PPG33','PPG34','PPG35','PPG36','PPG37','PPG38','PPG39','PPG40','PPG41','PPG42','PPG43','PPG44','PPG45','PPG46',
#                         'PPG47','PPG48','PPG49','PPG50','PPG51','PPG52','PPG53']
    

# ExcludeList220808_Ct_Osmo = ['PPG1','PPG2','PPG3','PPG4','PPG5','PPG6','PPG7','PPG8','PPG9','PPG10','PPG11','PPG12','PPG13','PPG14','PPG15','PPG16',
#                         'PPG17','PPG18','PPG19','PPG20','PPG21','PPG22','PPG23','PPG24','PPG25','PPG27','PPG28','PPG29','PPG30','PPG31',
#                         'PPG32','PPG33','PPG34','PPG35','PPG36','PPG37','PPG38','PPG39','PPG41','PPG42','PPG43','PPGA']

    
# ExcludeList220808_100mM_Osmo = ['PPG1','PPG2','PPG3','PPG4','PPG5','PPG6','PPG7','PPG8','PPG9','PPG10','PPG11','PPG12','PPG13','PPG14','PPG15','PPG16',
#                         'PPG17','PPG19','PPG20','PPG21','PPG22','PPG23','PPG24','PPG25','PPG26','PPG27','PPG28','PPG29','PPG31',
#                         'PPG32','PPG33','PPG34','PPG35','PPG36','PPG37','PPG38','PPG39','PPG40','PPG41','PPG42','PPG43','PPG44','PPG45','PPG46',
#                         'PPG47','PPG48','PPG49','PPG50','PPG51','PPG52','PPG53'] #

    
# ExcludeList220928_Ct = ['PPG1','PPG2','PPG3','PPG4','PPG5','PPG6','PPG7','PPG8','PPG9','PPG10','PPG11','PPG12','PPG13','PPG14','PPG15','PPG16',
#                         'PPG17','PPG18','PPG19','PPG20','PPG21','PPG22','PPG23','PPG24','PPG25','PPG26','PPG27','PPG28','PPG29','PPG30',
#                         'PPG32','PPG33','PPG34','PPG35','PPG36','PPG37','PPG38','PPG39','PPG40','PPG41','PPG42','PPG43','PPG44','PPG45','PPG46',
#                         'PPG47','PPG48','PPG49','PPG50']

    
# ExcludeList220928_100mM = ['PPG1','PPG2','PPG3','PPG4','PPG5','PPG7','PPG8','PPG9','PPG10','PPG11','PPG12','PPG13','PPG14','PPG15','PPG16',
#                         'PPG17','PPG18','PPG19','PPG20','PPG21','PPG22','PPG23','PPG24','PPG25','PPG26','PPG27','PPG28','PPG29','PPG30','PPG31',
#                         'PPG32','PPG33','PPG34','PPG35','PPG36','PPG37','PPG38','PPG39','PPG40','PPG41','PPG42','PPG43','PPG44','PPG45','PPG46']
    

# ExcludeList220928_Ct_Osmo = ['PPG1','PPG2','PPG3','PPG4','PPG5','PPG6','PPG7','PPG8','PPG9','PPG10','PPG11','PPG12','PPG13','PPG14','PPG15','PPG16',
#                         'PPG17','PPG18','PPG19','PPG20','PPG21','PPG22','PPG23','PPG24','PPG25','PPG26','PPG27','PPG28','PPG29','PPG30','PPG31',
#                         'PPG32','PPG34','PPG35','PPG36','PPG37','PPG38','PPG39','PPG40','PPG41','PPG42','PPG43','PPG44','PPG45','PPG46',
#                         'PPG47','PPG48','PPG49','PPG50','PPGA','PPGB','PPGC']

    
# ExcludeList220928_100mM_Osmo = ['PPG1','PPG2','PPG3','PPG4','PPG5','PPG7','PPG8','PPG9','PPG10','PPG11','PPG12','PPG13','PPG14','PPG15','PPG16',
#                         'PPG17','PPG18','PPG19','PPG20','PPG21','PPG22','PPG23','PPG24','PPG25','PPG26','PPG27','PPG28','PPG29','PPG30','PPG31',
#                         'PPG32','PPG33','PPG35','PPG36','PPG37','PPG38','PPG39','PPG40','PPG41','PPG42','PPG43','PPG44','PPG45','PPG46','PPGA','PPGB',
#                         'PPGC']


# ExcludeList221004_Ct_Osmo = ['PPG1','PPG2','PPG3','PPG4','PPG5','PPG6','PPG7','PPG8','PPG9','PPG10','PPG11','PPG12','PPG13','PPG14','PPG15','PPG16',
#                         'PPG17','PPG18','PPG19','PPG20','PPG21','PPG22','PPG23','PPG24','PPG25','PPG26','PPG27','PPG28','PPG29','PPG30','PPG31',
#                         'PPG32','PPG33','PPG34','PPG35','PPG36','PPG37','PPG38','PPG39','PPG40','PPG41','PPG42','PPG43','PPG44','PPG45']

# ExcludeList221004_Pase10U_Osmo = ['PPG1','PPG2','PPG3','PPG4','PPG5','PPG6','PPG7','PPG8','PPG10','PPG11','PPG12','PPG13','PPG14','PPG15','PPG16',
#                     'PPG17','PPG18','PPG19','PPG20','PPG21','PPG22','PPG23','PPG24','PPG25','PPG26','PPG27','PPG28','PPG29','PPG30','PPG31',
#                     'PPG32','PPG33','PPG34','PPG35','PPG36','PPG37','PPG38','PPG39','PPG40','PPG41','PPG42','PPG43','PPG44']

print('Contour data choice made.')