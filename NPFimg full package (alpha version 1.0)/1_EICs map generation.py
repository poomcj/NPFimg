'''
====================================================================================================================================================================================
1. Convert Raw Data to Image
2. Generate the Image and Increase data (Augmentation) by using noise variation
Ref : https://c13s.wordpress.com/2010/04/13/add-noise-to-data/
3. Save Image and Define Image Property and Quality
====================================================================================================================================================================================
Chaiyanut Jirayupat
'''

###################################################################################################################################################################
import os
import sys
import time
import shutil
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from gcmstools.filetypes import AiaFile
from matplotlib.colors import PowerNorm
from skimage import io, color, img_as_ubyte
###################################################################################################################################################################

###################################################################################################################################################################
#Parameter definition
Mtrix =[]
Nameset = []
###################################################################################################################################################################

###################################################################################################################################################################
'''
============================================================
Location of Raw data (.CDF) / *** Need modify ***
============================================================
'''
path = r'\\192.168.200.4\share\usr\Nagashima\Poom data\Testing data for AISIN\Aroma1+3additive molecules'
Dpi_calibration_img_path = 'DPI_Calibration/W100_H100_DPI100.jpg'

print(path)

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

def massageCheck(massage):
    while True:
        try: Quesion_check = int(input(massage))
        except ValueError: print("ERROR, Try again."); continue
        if Quesion_check>1: print("ERROR, Try again."); continue
        else: return Quesion_check; break

def AiaReader(filepath,name):
    Filename = os.path.join(filepath, name)
    Data = AiaFile(Filename)
    TwoDmap = Data.intensity[:]
    return TwoDmap

def noise_func(a,b,c):
    return a * b * c/100

def makeNoise(img_matrix, noise_mean, noise_std, noise_i):
    n1 = np.random.normal(noise_mean, noise_std, size=np.shape(img_matrix))
    n2 = noise_func(n1, img_matrix, noise_i)
    TwoDmap = (img_matrix + n2)
    TwoDmap_with_noise = TwoDmap - np.min(TwoDmap)
    return TwoDmap_with_noise

def forceAspect(ax,aspect):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def imgReader(Path):
    image = io.imread(Path)
    image = img_as_ubyte(color.rgb2gray(image))
    return image

def saveImg(image, gamma, dpis, dirc, name, noise_i, num):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, interpolation='bilinear', cmap='gray', origin='lower',
              extent=[0, int(image.shape[1]), 0, int(image.shape[0])], #vmax=np.max(image),
              norm=PowerNorm(gamma))
    forceAspect(ax, aspect=float(4/3))  # float(4/3))
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # To save img data
    fname = '/{}_noise_{}%_sample_#{}.png'.format(name, noise_i, num)
    fig.savefig(dirc+fname, transparent=True, #frameon=False,
                bbox_inches='tight', pad_inches=0 , dpi=dpis)
    plt.cla; plt.clf;plt.close()
    return fname

def findmyDpi(std_img_path, image_final_resolution):
    i=0
    dpi_set = [10, 100, 250, 500, 1000]
    std_img = imgReader(std_img_path)
    gamma = 0; std_name = 'Standard Image for Find My DPI'
    noise = 'None'; sample_num = 'None'
    Dpi_dir = newFolder(std_name)
    my_dpi_set = []
    for dpi in dpi_set:
        std_img_dpi = saveImg(std_img, gamma, dpi, Dpi_dir, std_name, noise, sample_num)
        #print(std_img_dpi)
        new_img_dpi = imgReader(Dpi_dir+std_img_dpi)
        my_dpi_set = np.append(my_dpi_set, np.shape(new_img_dpi[0]))
        i+=1; print(printProgressBar(i, len(dpi_set)))
    find_my_dpi = np.polyfit(my_dpi_set, dpi_set, 1)
    new_dpi = (find_my_dpi[0] * image_final_resolution) + find_my_dpi[1]
    shutil.rmtree(Dpi_dir)
    return new_dpi
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
sample_name = input("Please Fill the Sample Name/Condition for Saving the Data : ")
dir_path = newFolder(sample_name)
print("Data were Saved at Directory : {} ".format(dir_path))
print("--------------------- Setting Parameter for Data Augmentation and Image Generation ---------------------")
Quesion_check = massageCheck("Do You Want to Set Up Noise by Manual(0) or Automatic(1) : ")
if Quesion_check == 0:
    Max_noise = inputNumber("The Highest Percentage of Noise(%) : ")
    Min_noise = inputNumber("The Lowest  Percentage of Noise(%) : ") #it should be zero.
    Step_noise = inputNumber("The Step for increasing Noise(%)   : ")
    std = inputNumber("The Standard Deviation (Spread or “Width”) of the Distribution : ")
    mean = inputNumber("The Mean (“centre”) of the Distribution : ")
    noise = np.arange(Min_noise, Max_noise + 1, Step_noise)
else: std = 1; mean = 0 ; Max_noise = 11; Min_noise = 0; Step_noise = 1; noise = np.arange(Min_noise,Max_noise,Step_noise);
print("Noise Variation Set : ", noise)
print("Number of Generated image/Sample : ", int(noise.shape[0]))
print("--------------------------------------------------------------------------------------------------------")
###################################################################################################################################################################

###################################################################################################################################################################
print("------------------------------- Setting Parameter for Image Filtering ----------------------------------")
Quesion_check = massageCheck("Do You Want to Active Image Filter or Not? (Yes = 1 or No = 0) : ")
if Quesion_check == 1:
    Quesion_check2 = massageCheck("Do You Want to Set Up Img Filter Gamma by Manual(0) or Automatic(1) : ")
    if Quesion_check2 == 0: gamma = inputNumber("Input Gamma Value : ")
    else : gamma = 0.3; print("Standard Gamma for Image Filtering = ",gamma)
else: gamma = 0.1; print("Gamma  = ",gamma)
print("--------------------------------------------------------------------------------------------------------")
###################################################################################################################################################################

###################################################################################################################################################################
print("------------------------------- Setting Your Image Resolution / DPI ! ----------------------------------")
Quesion_check = massageCheck("Do You Want to Set Up Image Resolution by Manual(0) or Automatic(1) : ")
if Quesion_check ==1: image_resolution_w = 610; print('The Image Resolution (height x width) : {}x{}'.format(int(image_resolution_w*3/4), image_resolution_w))
else:
    image_resolution_w = inputNumber("The Width of Image Resolution (The Image Ratio (hXw is fixed equal 3:4): ")
    if image_resolution_w != 610 :
        print("--------------------------------------------------------------------------------------------------------") #This part only for alpha tester who use the Aroma sample.
        print("--------------------------------------------------------------------------------------------------------") #This part only for alpha tester who use the Aroma sample.
        print("------------------------------- Warning ! Warning ! Warning ! Warning ----------------------------------") #This part only for alpha tester who use the Aroma sample.
        print("------------------------------- Warning ! Warning ! Warning ! Warning ----------------------------------") #This part only for alpha tester who use the Aroma sample.
        print("------------------------------- Warning ! Warning ! Warning ! Warning ----------------------------------") #This part only for alpha tester who use the Aroma sample.
        print("--------------------------------------------------------------------------------------------------------") #This part only for alpha tester who use the Aroma sample.
        print("--------------------------------------------------------------------------------------------------------") #This part only for alpha tester who use the Aroma sample.
        print("If The Width of Image Resolution is not Equal 610 Pixel,You must Manually Identify the Reference Peak of") #This part only for alpha tester who use the Aroma sample.
        print("Testing Sample : Aroma#1 and Aroma#1 + 3 additive molecules")
        print("--------------------------------------------------------------------------------------------------------") #This part only for alpha tester who use the Aroma sample.
        print("--------------------------------------------------------------------------------------------------------") #This part only for alpha tester who use the Aroma sample.
        print("------------------------------- Warning ! Warning ! Warning ! Warning ----------------------------------") #This part only for alpha tester who use the Aroma sample.
        print("------------------------------- Warning ! Warning ! Warning ! Warning ----------------------------------") #This part only for alpha tester who use the Aroma sample.
        print("------------------------------- Warning ! Warning ! Warning ! Warning ----------------------------------") #This part only for alpha tester who use the Aroma sample.
        print("--------------------------------------------------------------------------------------------------------") #This part only for alpha tester who use the Aroma sample.
        print("--------------------------------------------------------------------------------------------------------") #This part only for alpha tester who use the Aroma sample.
# *** note the final image resolusion is 450x600 (wxh)
print("DPI Calibration Start!!!")
my_dpi = findmyDpi(Dpi_calibration_img_path, image_resolution_w)
print("--------------------------------------------------------------------------------------------------------")
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
print("-------------------------------- !!!!!!!Data Generation Start!!!!!!! -----------------------------------")
for file in os.listdir(path):
    if file.endswith('.CDF'): Nameset.append(file)
for k in range(len(Nameset)):
    Mtrix = AiaReader(path,Nameset[k])
    for i in range(len(noise)):
        Mtrix_with_noise = makeNoise(Mtrix, mean, std, noise[i])
        Sample = saveImg(Mtrix_with_noise, gamma, my_dpi, dir_path, sample_name, noise[i], k)
    print(printProgressBar(k+1, len(Nameset),))
###################################################################################################################################################################

###################################################################################################################################################################
print("========================================================================================================")
sample_condition = f'{sample_name}', f'{Max_noise}', f'{Min_noise}', f'{Step_noise}', \
                   f'{std}', f'{mean}', f'{noise.shape[0]}', f'{gamma}', \
                   f'{int(image_resolution_w*3/4)} x {int(image_resolution_w)}', f'{my_dpi}'
sample_condition = pd.Series(sample_condition, index=['sample name', 'Max noise', 'Min noise', 'Step noise',
                                                      'std', 'mean', 'number of augmentation data', 'gamma',
                                                      'image resolution', 'my dpi'],
                             name='Image Generation Report')
sample_condition.to_csv(dir_path+'/Image Generation Report.txt')
print(sample_condition)
print("========================================================================================================")
###################################################################################################################################################################

###################################################################################################################################################################
print("-------------------------------- !!!!!!!Data is Saved!!!!!!! -----------------------------------")
print("========================================================================================================")
print("Program End!!!")
print("========================================================================================================")
###################################################################################################################################################################
