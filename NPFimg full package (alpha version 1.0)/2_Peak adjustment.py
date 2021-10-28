'''
====================================================================================================================================================================================
1. Image alignment
2. Save the aligned image and define image resolution
Ref : https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_dog
====================================================================================================================================================================================
Chaiyanut Jirayupat
'''

###################################################################################################################################################################
import os
import sys
import time
import matplotlib
import numpy as np
import pandas as pd
from math import sqrt
matplotlib.use("TkAgg")
from pathlib import Path
from skimage import feature
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte
###################################################################################################################################################################
#parameter define
Nameset = []
peak_total=[]
###################################################################################################################################################################

###################################################################################################################################################################
'''
============================================================
Location of 2D EICs Map(.png) / *** Need modify ***
============================================================
'''
# Read file
# location of sample folder
path = r'C:\Users\yanag\OneDrive\デスクトップ\Poom\NPFimg package for AISIN (alpha version 2.1)\Results\Aroma+3m' #
# location of reference image(.png)
path_ref = r'C:\Users\yanag\OneDrive\デスクトップ\Poom\NPFimg package for AISIN (alpha version 2.1)\Results\Aroma\Aroma_noise_0.0%_sample_#0.png' #
#Must to use the same reference image for all sample/condition
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

def printProgressBar(i,max):
    n_bar =10 #size of progress bar
    j= i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'#' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {'Running' + ' -------- '}")
    sys.stdout.flush()

def inputNumber(massage):
    while True:
        try: userInput = float(input(massage))
        except ValueError: print("ERROR, Try again."); continue
        else: return userInput; break

def massage_check(massage):
    while True:
        try: Quesion_check = int(input(massage))
        except ValueError: print("ERROR, Try again."); continue
        if Quesion_check>1: print("ERROR, Try again."); continue
        else: return Quesion_check; break

def imgPlot(path, title, active_plot):
    image = io.imread(path);
    image = img_as_ubyte(color.rgb2gray(image));
    print(image.shape)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='jet')
    ax.set_ylabel('Retention time')
    ax.set_xlabel('m/Z')
    ax.set_title(title)
    if active_plot == 1: plt.show()
    else: pass
    plt.cla; plt.clf;plt.close()
    print(title)
    return image


def blobDetection(image, title, active_plot, y_win_ref, x_win_ref, h, w, min_sigmas, max_sigmas, thresholds, overlaps):
    # peak detection of reference peak (reference image)
    peak = []
    image = image[y_win_ref:y_win_ref + h, x_win_ref:x_win_ref + w]
    blobs_dog = feature.blob_dog(image, min_sigma=min_sigmas, max_sigma=max_sigmas, threshold=thresholds, overlap=overlaps);
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    fig, ax = plt.subplots();
    ax.imshow(image, cmap='jet')
    ax.set_ylabel('Retention time')
    ax.set_xlabel('m/Z')
    ax.set_title('Reference peak of {}'.format(title))
    for i, blob in enumerate(blobs_dog):
        y, x, r = blob;
        x = int(x);
        y = int(y);
        z = image[y, x];
        c = plt.Circle((x, y), r, color='white', linewidth=2, fill=False)
        ax.text(x, y, (y, x), size=20, horizontalalignment='right', color='red');
        ax.add_patch(c)
        if i == 0: peak = [(y, x, z)]
        else: peak = np.vstack((peak, [(y, x, z)]))
    peak = sorted(peak, key=lambda tup: tup[2]);
    ymax = peak[len(peak) - 1][0];
    xmax = peak[len(peak) - 1][1];
    zmax = peak[len(peak) - 1][2]
    print("{} ----- Ref Peak Position y = {}, x = {} and Intensity z = {}".format(title, ymax, xmax, zmax));
    if active_plot == 1: plt.show()
    else: pass
    plt.cla; plt.clf;plt.close()
    return ymax, xmax, zmax

def posAlign(ref_pos, sam_pos, win_ref, scan_window_size):
    new_ref_pos = win_ref
    i=0
    while ref_pos > sam_pos : new_ref_pos -= 1; sam_pos += 1; i += 1
    else:
        while ref_pos < sam_pos: new_ref_pos += 1; sam_pos -= 1; i += 1
        else: pass
    if i > scan_window_size : print("Warning Massage : The Aligned Position is Over than The Setting Value --- Please Carefully Check Your Data !!!!! ---")
    adj_para = new_ref_pos - win_ref
    return  adj_para
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

for file in os.listdir(path):
    if file.endswith(".png"): Nameset.append(file)
sample_name = Nameset[0].split("_")
sample_name = 'Alignment image for {}'.format(sample_name[0])
dir_path = newFolder(sample_name)
print("Sample Name/Condition : {} ".format(sample_name))
print("Data were Saved at Directory : {} ".format(dir_path))

print("--------------------- Setting Peak Detection (Blob Detection) Parameter for Position Alignment ---------------------")
Quesion_check = massage_check("Do You Want to Set Up Peak Detection for Detecting the Reference Peak by Manual(0) or Automatic(1) : ")
if Quesion_check == 0:
    min_sigmas = inputNumber("The Minimum Standard Deviation for Gaussian Kernel (Min_sigmas) :")
    max_sigmas = inputNumber("The Maximum Standard Deviation for Gaussian Kernel (Max_sigmas) :")
    thresholds = inputNumber("The Absolute Lower Bound for Scale Space Maxima (Thresholds)    :")
    overlaps = inputNumber("Overlaps Value Must Between 0 and 1 :")
else:
    min_sigmas = 1
    max_sigmas = 5
    thresholds = 0.01
    overlaps = 1
print("Blob Detection Parameter, Min_sigmas : {}, Max_sigmas = {}, Thresholds = {}, and Overlaps = {} ".format(min_sigmas, max_sigmas, thresholds, overlaps))
print("--------------------------------------------------------------------------------------------------------------------")
###################################################################################################################################################################

###################################################################################################################################################################
print("------------------------------------ Setting Parameter for Position Alignment --------------------------------------")
Quesion_check = massage_check("Do You Want to Set Up Position Alignment by Manual(0) or Automatic(1) : ")
if Quesion_check == 0:
    y_win_ref = inputNumber("The Lowest Position of Window on Y Axis for Focusing the Reference Peak : ")
    x_win_ref = inputNumber("The Lowest Position of Window on X Axis for Focusing the Reference Peak : ")
    h = inputNumber("The Height of Window on Y Axis for Focusing the Reference Peak : ")
    w = inputNumber("The Width of Window on X Axis for Focusing the Reference Peak  : ")
    scan_window_size = inputNumber("A Miximum Size on both of X and Y Axis to Move the Window for Image Alignment) : ")
    Img_ref = io.imread(path_ref); GreyImg_ref = img_as_ubyte(color.rgb2gray(Img_ref));
    print("Original Image Metrix Shape : ", GreyImg_ref.shape)
    h_adj_full = inputNumber("The Height of Window on Y Axis for Adjusting the Position of EICs Map (New Image Metrix Size Should be Smaller than Original Image Metrix Size) : ")
    w_adj_full = inputNumber("The Width of Window on X Axis for Adjusting the Position of EICs Map (New Image Metrix Size) : ")
    position_adj_para = inputNumber("Position Alignment Window Size (A Miximum Size on both of X and Y Axis to Move the Window for Image Alignment on Full EICs Map) : ")
else:
    #define the position for reference peak
    y_win_ref = 380 #lowest position on y axis for focusing of reference peak
    x_win_ref = 120 #lowest position on x axis for focusing of reference peak
    #define the window size for focusing reference peak
    h = 25 #size of window on y axis for focusing the reference peak
    w = 25 #size of window on x axis for focusing the reference peak
    # define the full EICs image size
    h_adj_full = 450 #size of window on y axis for adjusting the position of EICs map
    w_adj_full = 600 #size of window on x axis for adjusting the position of EICs map
    scan_window_size = 50
    position_adj_para = 5
    print("--------------------------------------------------------------------------------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("--------------------------------------------------------------------------------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("------------------------------- Warning ! Warning ! Warning ! Warning ----------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("------------------------------- Warning ! Warning ! Warning ! Warning ----------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("------------------------------- Warning ! Warning ! Warning ! Warning ----------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("--------------------------------------------------------------------------------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("--------------------------------------------------------------------------------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("        These Parameter use only for Testing Sample : Aroma#1 and Aroma#1 + 3 additive molecules        ")  # This part only for alpha tester who use the Aroma sample.
    print("--------------------------------------------------------------------------------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("--------------------------------------------------------------------------------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("----------------------------- Reminder ! Reminder ! Reminder ! Reminder --------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("--------------------------------------------------------------------------------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("--------------------------------------------------------------------------------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("If The Width of Image Resolution is not Equal 600 Pixel,You must Manually Identify the Reference Peak of")  # This part only for alpha tester who use the Aroma sample.
    print("Testing Sample : Aroma#1 and Aroma#1 + 3 additive molecules")
    print("--------------------------------------------------------------------------------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("--------------------------------------------------------------------------------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("------------------------------- Warning ! Warning ! Warning ! Warning ----------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("------------------------------- Warning ! Warning ! Warning ! Warning ----------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("------------------------------- Warning ! Warning ! Warning ! Warning ----------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("--------------------------------------------------------------------------------------------------------")  # This part only for alpha tester who use the Aroma sample.
    print("--------------------------------------------------------------------------------------------------------")  # This part only for alpha tester who use the Aroma sample.
Quesion_check = massage_check("Do You Want to Manually Check the Alignment Process or Not (Yes = 1 or No = 0) : ")
if Quesion_check == 1: active_plot = 1
elif Quesion_check == 0: active_plot = 0
print("--------------------------------------------------------------------------------------------------------------------")
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
for q in range(len(Nameset)):
    print("--------------------------------------------------------------------------------------------------------")
    print("------------------------------------ Sample #{} Start Alignment ----------------------------------------".format(q))
    ###################################################################################################################################################################
    # load reference image and focusing on the reference peak position
    if q == 0:
        title = "Reference Image Plot"
        ref_image = imgPlot(path_ref, title, active_plot)
        #peak detection of reference peak (reference image)
        title = "Reference Image"
        y_pos, x_pos, z_value = blobDetection(ref_image, title, active_plot, y_win_ref, x_win_ref, h, w, min_sigmas, max_sigmas, thresholds, overlaps)
    else: pass
    print(printProgressBar(q+1, len(Nameset),))
####################################################################################################################################################################

###################################################################################################################################################################
    title = "Sample Image Before Adjust Position"
    #print("Sample Name", Nameset[q], q * 100 / len(Nameset))
    #peak detection of reference peak (input sample image)
    Filename = os.path.join(path, Nameset[q])
    Img_sam = io.imread(Filename)
    Img_sam = img_as_ubyte(color.rgb2gray(Img_sam));
    y_pos_sam, x_pos_sam, z_value_sam = blobDetection(Img_sam, title, active_plot, y_win_ref, x_win_ref, h, w, min_sigmas, max_sigmas, thresholds, overlaps)
###################################################################################################################################################################

###################################################################################################################################################################
    y_adj_para = posAlign(y_pos, y_pos_sam, y_win_ref, scan_window_size)
    x_adj_para = posAlign(x_pos, x_pos_sam, x_win_ref, scan_window_size)
    print("New Position Image Adjusted Parameter = ", y_adj_para, x_adj_para)
###################################################################################################################################################################

###################################################################################################################################################################
    title = 'Sample Image After Adjust Position '
    y_adjusted_pos = y_adj_para + y_win_ref
    x_adjusted_pos = x_adj_para + x_win_ref
    y_pos_sam_adj, x_pos_sam_adj, z_value_sam_adj = blobDetection(Img_sam, title, active_plot, y_adjusted_pos, x_adjusted_pos, h, w, min_sigmas, max_sigmas, thresholds, overlaps)
###################################################################################################################################################################

###################################################################################################################################################################
    # adjusted full EICs image
    GreyImg_full = Img_sam[position_adj_para + y_adj_para: position_adj_para + y_adj_para + h_adj_full, position_adj_para + x_adj_para:position_adj_para + x_adj_para + w_adj_full]
    print("Adjusted Full EICs Map - Ref Peak Position y = {}, x = {} and Intensity z = {}".format(y_pos_sam_adj, x_pos_sam_adj, z_value_sam_adj));
    print('Adjusted Full EICs Map Shape : ',GreyImg_full.shape)
    if active_plot == 1 and q == 0:
        fig4, ax4 = plt.subplots();
        ax4.imshow(GreyImg_full, cmap='jet')
        ax4.set_ylabel('Retention time')
        ax4.set_xlabel('m/Z')
        ax4.set_title('Adjusted full EICs map')
        plt.show()
    else: pass
    #To save img data
    plt.imsave(dir_path+'/{}_adj.png'.format(Path(Nameset[q]).stem), GreyImg_full, cmap='gray')
    print("------------------------------------ EICs Map was Adjusted & Save --------------------------------------")
    print("--------------------------------------------------------------------------------------------------------")
###################################################################################################################################################################

###################################################################################################################################################################
print("========================================================================================================")
sample_condition = f'{sample_name}', f'{min_sigmas}', f'{max_sigmas}', f'{thresholds}', f'{overlaps}', f'{y_win_ref}', f'{x_win_ref}', f'{h}', \
                   f'{w}', f'{h_adj_full}', f'{w_adj_full}', f'{h_adj_full} x {w_adj_full}', f'{scan_window_size}', f'{position_adj_para}',
sample_condition = pd.Series(sample_condition, index=['sample name', 'min_sigmas', 'max_sigmas', 'thresholds',
                                                      'overlaps', 'y_win_ref', 'x_win_ref', 'h', 'w',
                                                      'h_adj_full', 'w_adj_full', 'final resolution', 'Miximum Size of Aligned Position', 'position_adj_para'],
                             name='Alignment Report')
sample_condition.to_csv(dir_path+'/Alignment Report.txt')
print(sample_condition)
print("========================================================================================================")
###################################################################################################################################################################

###################################################################################################################################################################
print("========================================================================================================")
print("Program End!!!")
print("========================================================================================================")
###################################################################################################################################################################