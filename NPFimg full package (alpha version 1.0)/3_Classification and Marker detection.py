'''
=================================================================
1. Classification and Marker detection
2. Generate Feature Map and save
=================================================================

Chaiyanut Jirayupat
'''

###################################################################################################################################################################
import os
import sys
import time
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("TkAgg")
from sklearn import metrics
from matplotlib import colors
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
###################################################################################################################################################################
#Parameter definition
ans = []
Nameset = []
img_set = []
image_set = []
test_sizes = 0.33
feature_score = []
np.random.seed(1234)
seconds = time.time()
###################################################################################################################################################################

###################################################################################################################################################################
'''
============================================================
Location of 2D EICs Map(.png) / *** Need modify ***
============================================================
'''
# Read file
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

def imgPlot(path, Nameset_k, k):
    #print('Tag# :', k, 'Sample name :', Nameset[k])
    Filename = os.path.join(path,Nameset_k)
    image = io.imread(Filename)
    image = img_as_ubyte(color.rgb2gray(image))
    image_flat = image.flatten()
    image_set.append(image_flat)
    return image_set, image.shape

def scoreHist(feature):
    fig, axs = plt.subplots(1, 1,figsize=(10, 7), tight_layout=False)
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        axs.spines[s].set_visible(False)
    # Remove x, y ticks
    axs.xaxis.set_ticks_position('none')
    axs.yaxis.set_ticks_position('none')
    # Add padding between axes and labels
    axs.xaxis.set_tick_params(pad=5)
    axs.yaxis.set_tick_params(pad=10)
    # Add x, y gridlines
    axs.grid(b=True, color='grey',linestyle='-.', linewidth=0.5,alpha=0.6)
    # Creating histogram
    N, bins, patches = axs.hist(feature, bins=100)
    # Setting color
    fracs = ((N ** (1 / 5)) / N.max())
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    # Adding extra features
    plt.xlabel("Feature Score")
    plt.ylabel("Frequency")
    plt.title("Histogram of Feature Score")
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.show()

def FeatureMapPlot(Quesion_check, feature):
    while np.absolute(int(Quesion_check)) == 1 :
            threshold = inputNumber("Set Up Feature Score Threshold (Standard Threshold = 0.00005) : ")
            fig, ax = plt.subplots()
            ax.imshow(feature, cmap='magma', interpolation='bilinear', aspect='auto', vmax=np.max(feature), vmin=threshold)
            ax.set_title('2D Feature map')
            ax.set_ylabel('Retention time(image scale)')
            ax.set_xlabel('m/Z')
            plt.show()
            Quesion_check = massage_check("Do You Want to Modify Feature Score Threshold Again or Not (Yes = 1 or No = 0) : ")
            if Quesion_check == 0: return threshold; break
            else: continue
    else:
        threshold = 0.00005
        print("Feature Score Threshold is Automatically Set Up")
        return threshold
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
sample_bname = Nameset[len(Nameset)-1].split("_")
sample_name = 'Feature Map of {} vs {}'.format(sample_aname[0], sample_bname[0])
dir_path = newFolder(sample_name)
print("Sample Name/Condition : {} ".format(sample_name))
print("Feature Map was Saved at Directory : {} ".format(dir_path))
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
print("-------------------------------- Setting Parameter for Answer and Feature --------------------------------")
###################################################################################################################################################################

###################################################################################################################################################################
#Define Ans
ans = np.ravel(label_binarize(ans, classes=[0, 1]))
#Nameset = Nameset[::-1]
for k in range(len(Nameset)):
    if k < (ans[np.where(ans == 0)].size) : path = path_a
    else: path = path_b
    img_set, img_shape = imgPlot(path, Nameset[k], k)
    if (k+1)%10 == 0:print(printProgressBar(k+1, len(Nameset)))
feature_set = np.reshape(img_set, [len(ans),int(img_shape[1])*int(img_shape[0])])
print('Ans shape :', feature_set.shape[0], 'Feature shape :', feature_set.shape[1])
print("--------------------------------------------------------------------------------------------------------")
###################################################################################################################################################################

###################################################################################################################################################################
# generate training and testing data
X_train, X_validation, Y_train, Y_validation = train_test_split(feature_set, ans, test_size=test_sizes, random_state=44)
model = LogisticRegression()
# fit the model
model.fit(X_train,Y_train)
y_pred = model.predict(X_validation)
print("Testing Accuracy : ",metrics.accuracy_score(Y_validation, y_pred))
# summarize feature importance for logistic model
sec = time.time()
print("Total analysis time = {} min".format((sec-seconds)/60))
###################################################################################################################################################################

###################################################################################################################################################################
importance = model.coef_[0]
for i, v in enumerate(importance):
    feature_score.append(v)
scoreHist(feature_score)
###################################################################################################################################################################

###################################################################################################################################################################
feature_map = np.reshape(feature_score,[int(img_shape[0]),int(img_shape[1])])
print('Maximum score : {}, Minimum score {}'.format(np.max(feature_map),np.min(feature_map)))
Quesion_check = massage_check("Do You Want to Manually Set Up Feature Score Threshold or Not (Yes = 1 or No = 0) :")
feature_th = FeatureMapPlot(Quesion_check, feature_map)
print("Feature Score Threshold = ", feature_th)
# To save img data
plt.imsave(dir_path+'/{}.png'.format(sample_name), feature_map, cmap='magma', vmax=np.max(feature_map), vmin=feature_th)
###################################################################################################################################################################

###################################################################################################################################################################
print("========================================================================================================")
sample_condition = f'{sample_aname[0]}', f'{sample_bname[0]}', f'{ans[np.where(ans == 0)].size}', f'{ans[np.where(ans == 1)].size}', \
                   f'{test_sizes}', f'{model}', f'{metrics.accuracy_score(Y_validation, y_pred)}', f'{(sec-seconds)/60}', \
                   f'{feature_th}', f'{np.max(feature_map)}', f'{np.min(feature_map)}', f'{feature_set.shape[1]}'

sample_condition = pd.Series(sample_condition, index=['sample a name', 'sample b name', 'sample a num', 'sample b num',
                                                      'testing size', 'analysis model', 'classification accuracy', 'total analysis time (min)',
                                                      'Feature Score Threshold', 'Maximum score', 'Minimum score', 'Feature num'],
                             name='Classification and Feature Map Report')
sample_condition.to_csv(dir_path+'/Classification and Feature Map Report.txt')
print(sample_condition)
print("========================================================================================================")
###################################################################################################################################################################

###################################################################################################################################################################
print("-------------------------------- !!!!!!!Data is Saved!!!!!!! -----------------------------------")
print("========================================================================================================")
print("Program End!!!")
print("========================================================================================================")
###################################################################################################################################################################