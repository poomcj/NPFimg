# -*- coding: utf-8 -*-
"""
Extract the raw signal from image data
*** need updated !!!!!!!!!!!
"""


import os
import sys
import time
import matplotlib
import numpy as np
import pandas as pd
from math import sqrt
matplotlib.use("TkAgg")
from skimage import feature
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte
from sklearn.preprocessing import label_binarize
from scipy.interpolate import make_interp_spline, BSpline
###################################################################################################################################################################

###################################################################################################################################################################
peak=[]
ans = []
Nameset = []
###################################################################################################################################################################

###################################################################################################################################################################
'''
============================================================
Location of 2D EICs Map(.png) / *** Need modify ***
============================================================
'''
# Read feature map
path_feature = r'C:\Users\yanag\OneDrive\デスクトップ\Poom\NPFimg package for AISIN (alpha version 2.1)\Results\Feature Map of Aroma vs Aroma+3m/Feature Map of Aroma vs Aroma+3m.png'
#File path
path_a= r'C:\Users\yanag\OneDrive\デスクトップ\Poom\NPFimg package for AISIN (alpha version 2.1)\Results\Alignment image for Aroma' # Data set1
path_b= r'C:\Users\yanag\OneDrive\デスクトップ\Poom\NPFimg package for AISIN (alpha version 2.1)\Results\Alignment image for Aroma+3m' # Data set2
path_set = [path_a, path_b]
'''
============================================================
'''
###################################################################################################################################################################

###################################################################################################################################################################

#//////////////////////////////////////////////////////////////////////////// Main Function ///////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////// Main Function ///////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////// Main Function ///////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////// Main Function ///////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////// Main Function ///////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////// Main Function ///////////////////////////////////////////////////////////////////////

###################################################################################################################################################################

###################################################################################################################################################################
def newFolder(folder_name):
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/{}'.format(folder_name))
    try:os.mkdir(results_dir)
    except OSError as error:print(error); time.sleep(0.1)
    return results_dir

def inputNumber(massage):
    while True:
        try: userInput = int(input(massage))
        except ValueError: print("ERROR, Try again."); continue
        else: return userInput; break

def massage_check(massage):
    while True:
        try: Quesion_check = int(input(massage))
        except ValueError: print("ERROR, Try again."); continue
        if Quesion_check>1: print("ERROR, Try again."); continue
        else: return Quesion_check; break

def blobDetection(image, min_sigmas, max_sigmas, thresholds, overlaps):
    # peak detection of reference peak (reference image)
    peak = []
    blobs_dog = feature.blob_dog(image, min_sigma=min_sigmas, max_sigma=max_sigmas, threshold=thresholds, overlap=overlaps);
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    fig, ax = plt.subplots();
    ax.imshow(image, cmap='magma')
    ax.set_ylabel('Retention time')
    ax.set_xlabel('m/Z')
    ax.set_title('Feature Map')
    for i, blob in enumerate(blobs_dog):
        y, x, r = blob;
        x = int(x);
        y = int(y);
        z = image[y, x];
        c = plt.Circle((x, y), r, color='white', linewidth=2, fill=False)
        ax.text(x, y, (y, x), size=20, horizontalalignment='right', color='red');
        ax.add_patch(c)
        if i == 0: peak = [(y, x, r)]; score = z
        else: peak = np.vstack((peak, [(y, x,r)])); score = np.vstack((score, z))
    RT_set = ConverYtoRT(peak[:, 0])
    MZ_set = ConverXtoMZ(peak[:, 1])
    peak = np.column_stack((RT_set, MZ_set, peak))
    plt.show()
    plt.cla; plt.clf;plt.close()
    return peak

###################################################################################################################################################################
def ConverYtoRT(pos_Y):
    RT = ((-0.0485)*(pos_Y)) + 25.021#((-0.0497)*(pos_Y+50)) + 25.439
    return (RT)

def ConverXtoMZ(pos_X):
    MZ = (0.4395*(pos_X)) + 36.396
    return (MZ)
###################################################################################################################################################################

def MetabolicMapPlot(Quesion_check, array, ans, Dy, peak , sample_aname, sample_bname):
    while np.absolute(int(Quesion_check)) == 1:
        peak_num = inputNumber("Which peak do you want to plot(number)? : ")
        # plot raw EICs spectrum
        x = np.arange(Dy)
        x_rt = peak[peak_num, 0]
        x = (x - int(x_rt))
        fig, ax = plt.subplots()
        ax.set_title('Image intensity of all sample')
        ax.set_ylabel('Intensity (image scale)')
        ax.set_xlabel('Retention time (pixel)')
        ax.set_ylim(0, 255)
        #ax.set_xlim(0,int(len(x)))
        # Select peak to plot
        print('Selected peak = peak number {}'.format(peak_num))
        for i in range(len(Nameset)):
            yl = i * Dy
            yr = yl + Dy
            if i < (ans[np.where(ans == 0)].size):
                y = array[yl:yr, peak_num]
                y = y - y.min()
                x_new = np.linspace(0, x.max(), int(len(x)))
                smt = make_interp_spline(x, y, k=deg_smt)
                y_smooth = smt(x_new)
                if i == 0: ax.plot(x_new, y_smooth, color='tab:blue', label=sample_aname[0])
                else: ax.plot(x_new, y_smooth, color='tab:blue', )
            else:
                y = array[yl:yr, peak_num]
                y = y - y.min()
                x_new = np.linspace(0, x.max(), int(len(x)))
                smt = make_interp_spline(x, y, k=deg_smt)
                y_smooth = smt(x_new)
                if i == int(len(Nameset) / 2) + 1: ax.plot(x_new, y_smooth, color='tab:orange', label=sample_bname[0])
                else: ax.plot(x_new, y_smooth, color='tab:orange', )
        ax.legend()
        plt.show()
        Quesion_check = massage_check("Do You Want to Plot Another Peak or Not? (Yes = 1 or No = 0) :")
        if Quesion_check == 0: break
        else: continue
    else: pass
###################################################################################################################################################################

###################################################################################################################################################################

#///////////////////////////////////////////////////////////////////////////// Input Parameter ////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////////// Input Parameter ////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////////// Input Parameter ////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////////// Input Parameter ////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////////// Input Parameter ////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////////// Input Parameter ////////////////////////////////////////////////////////////////////

###################################################################################################################################################################

###################################################################################################################################################################
print("========================================================================================================")
print("Program Start!!!")
print("========================================================================================================")

for a, path in enumerate(path_set):
    for file in os.listdir(path):
        if file.endswith('.png'):
            Nameset.append(file)
            ans.append(a)

sample_aname = Nameset[0].split("_")
sample_bname = Nameset[len(Nameset) - 1].split("_")
sample_name = 'Metabolomics of {} vs {}'.format(sample_aname[0], sample_bname[0])
dir_path = newFolder(sample_name)
print("Sample Name/Condition : {} ".format(sample_name))
print("Metabolomics Data was Saved at Directory : {} ".format(dir_path))
###################################################################################################################################################################

###################################################################################################################################################################
print("--------------------- Setting Peak Detection (Blob Detection) Parameter for Bio-/Chemo-Marker Detection ---------------------")
Quesion_check = massage_check("Do You Want to Set Up Peak Detection for Detecting the Bio-/Chemo-Marker Peak by Manual(0) or Automatic(1) : ")
if Quesion_check == 0:
    min_sigmas = inputNumber("The Minimum Standard Deviation for Gaussian Kernel (Min_sigmas) :")
    max_sigmas = inputNumber("The Maximum Standard Deviation for Gaussian Kernel (Max_sigmas) :")
    thresholds = inputNumber("The Absolute Lower Bound for Scale Space Maxima (Thresholds)    :")
    overlaps = inputNumber("Overlaps Value Must Between 0 and 1 :")
else:
    min_sigmas = 1
    max_sigmas = 6
    thresholds = 0.05
    overlaps = 0.8
print("Blob Detection Parameter, Min_sigmas : {}, Max_sigmas = {}, Thresholds = {}, and Overlaps = {} ".format(min_sigmas, max_sigmas, thresholds, overlaps))
###################################################################################################################################################################

###################################################################################################################################################################
Quesion_check = massage_check("Do You Want to Set Up Window Size for Peak Identification by Manual(0) or Automatic(1) : ")
if Quesion_check == 0: win_size = inputNumber("Window Size (For Aroma Training Data Set = 4) : ")
else: win_size =4; print("Window Size :", win_size)

Quesion_check = massage_check("Do You Want to Smooth Peak by Manual(0) or Automatic(1) : ")
if Quesion_check == 0: deg_smt = inputNumber("B-spline Degree (Default is Cubic, k=3.) : ")
else: deg_smt =3; print("B-spline Degree :", deg_smt)
###################################################################################################################################################################

###################################################################################################################################################################

#/////////////////////////////////////////////////////////////////////////////// Main Code ////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////// Main Code ////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////// Main Code ////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////// Main Code ////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////// Main Code ////////////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////////////////// Main Code ////////////////////////////////////////////////////////////////////////

###################################################################################################################################################################

###################################################################################################################################################################
#Define Ans
ans = np.ravel(label_binarize(ans, classes=[0, 1]))
# identify marker
Feature_Img = io.imread(path_feature);
Feature_Img = img_as_ubyte(color.rgb2gray(Feature_Img));
print("2D Feature map shape :",Feature_Img.shape)
# peak detection of reference peak (reference image)
peak = blobDetection(Feature_Img, min_sigmas, max_sigmas, thresholds, overlaps)
print('Marker set shape :', peak.shape)
peak_labels = sorted(peak, key=lambda tup: tup[2], reverse=True)
peak_list = pd.DataFrame(peak_labels)
peak_list.columns = ['Retention time(min)', 'm/z', 'Y position', 'X position', 'Peak width']
print("----- Peak list -----")
print(peak_list)
peak_list.to_csv(dir_path+'/Marker Report.txt')
###################################################################################################################################################################

###################################################################################################################################################################
for k in range(len(Nameset)):
    #print('Sample name :',k , Nameset[k])
    image = []
    Meta_peak_group=[]
    if k < (ans[np.where(ans == 0)].size): path = path_a
    else: path = path_b
    Filename = os.path.join(path, Nameset[k])
    img = io.imread(Filename)
    image = img_as_ubyte(color.rgb2gray(img))
    y_img = peak[:, 2]
    x_img = peak[:, 3]
    RT_width = peak[:,4]
    Meta_peak_set=[]
    for pos in range(len(y_img)):
        RT_central = peak[pos,2]
        mZ = peak[pos,3]
        RT_radius = np.max(peak[:,4])
        RT_width_right = int(RT_central)+(RT_radius*2)
        RT_width_left = int(RT_central)-(RT_radius*8)
        RT_width = RT_width_right-RT_width_left
        Meta_peak_set = []
        #print('width',RT_width)
        Dy = int(round(RT_radius)*16) ##### y dimension #####
        for j in range(Dy):
            Meta_peak = []
            Meta_point = image[int(RT_width_left+j), int(mZ)]
            Meta_peak = np.append(Meta_peak,Meta_point)
            if j == 0:Meta_peak_set = Meta_peak
            else:Meta_peak_set = np.vstack((Meta_peak_set,Meta_peak))
        if pos == 0:Meta_peak_group = Meta_peak_set
        else:Meta_peak_group = np.hstack((Meta_peak_group, Meta_peak_set))
    if k == 0:Meta_peak_all = Meta_peak_group
    else:Meta_peak_all = np.vstack((Meta_peak_all, Meta_peak_group))
print('Retention time width (Dy) : {} pixel, Total peaks (Dx) : {}, Total samples (Dz) :{}'.format(Dy,len(y_img),len(Nameset)))
###################################################################################################################################################################

###################################################################################################################################################################
#save data to txt file
for m in range(len(y_img)):
    Meta_save=[]
    Meta_save = Meta_peak_all[:,m]
    Meta_save = np.reshape(Meta_save, (len(Nameset),Dy))
    Meta_save = np.transpose(Meta_save)
    #save data to csv
    Meta_save = pd.DataFrame(Meta_save)
    Meta_save.to_csv(dir_path + '/Metabolomic_data_Peak_#{}.txt'.format(m))
print("-------------------------------- !!!!!!!Data is Saved!!!!!!! -----------------------------------")
###################################################################################################################################################################

###################################################################################################################################################################
Quesion_check = massage_check("Do You Want to Plot Raw Data of Bio-/Chemo-Marker? (Yes = 1 or No = 0) : ")
MetabolicMapPlot(Quesion_check, Meta_peak_all, ans, Dy, peak , sample_aname, sample_bname)
###################################################################################################################################################################

###################################################################################################################################################################
print("========================================================================================================")
sample_condition = f'{sample_aname[0]}', f'{sample_bname[0]}', f'{ans[np.where(ans == 0)].size}', f'{ans[np.where(ans == 1)].size}', \
                   f'{min_sigmas}', f'{max_sigmas}', f'{thresholds}', \
                   f'{overlaps}', f'{win_size}', f'{deg_smt}', f'{len(y_img)}', f'{Dy}'

sample_condition = pd.Series(sample_condition, index=['sample a name', 'sample b name', 'sample a num', 'sample b num', 'min_sigmas', 'max_sigmas', 'thresholds',
                                                      'overlaps', 'win_size', 'deg_smt', 'total marker peak' , f'Retention time width'],
                             name='Metabolomics Report')
sample_condition.to_csv(dir_path+'/Metabolomics Report.txt')
print(sample_condition)
print("========================================================================================================")
###################################################################################################################################################################

###################################################################################################################################################################
print("========================================================================================================")
print("Program End!!!")
print("========================================================================================================")
###################################################################################################################################################################





























