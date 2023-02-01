#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
from pydicom import dcmread

from skimage.transform import resize
import os
import shutil

def read_meta_data(filename):
    dataset = pd.read_csv(filename)
    return dataset


# In[2]:


##Read CBIS_DDSM metadata from your local folder.

cbisddsm = read_meta_data("/home/careinfolab/StanfordMammogram/metadata.csv")


# In[3]:


## Read other CBIS-DDSM metadata files that contains benign and malignant labels. 
def MalignOrBenign(DistinctName):
    cal_case_train_set = read_meta_data("/home/careinfolab/StanfordMammogram/calc_case_description_train_set.csv")
    cal_case_test_set = read_meta_data("/home/careinfolab/StanfordMammogram/calc_case_description_test_set.csv")
    mass_case_train_set = read_meta_data("/home/careinfolab/StanfordMammogram/mass_case_description_test_set.csv")
    mass_case_test_set = read_meta_data("/home/careinfolab/StanfordMammogram/mass_case_description_train_set.csv")

    cal_case_train = cal_case_train_set[cal_case_train_set["image file path"].str.contains(DistinctName)]
    cal_case_test = cal_case_test_set[cal_case_test_set["image file path"].str.contains(DistinctName)]
    mass_case_train = mass_case_train_set[mass_case_train_set["image file path"].str.contains(DistinctName)]
    mass_case_test = mass_case_test_set[mass_case_test_set["image file path"].str.contains(DistinctName)]

    if cal_case_train.shape[0] > 0:
        label = cal_case_train['pathology'].values.astype('str')[0]
        return label

    if cal_case_test.shape[0] > 0:
        label = cal_case_test['pathology'].values.astype('str')[0]
        return label

    if mass_case_train.shape[0] > 0:
        label = mass_case_train['pathology'].values.astype('str')[0]
        return label

    if mass_case_test.shape[0] > 0:
        label = mass_case_test['pathology'].values.astype('str')[0]
        return label

    return "No_Label"


# In[4]:


## Method to convert from DICOM to PNGs
def ConvertFromDCMtoPNG(srcPath, dstFolderPath):
    im_dim_x = 800
    im_dim_y = 600

    im = pydicom.dcmread(srcPath)
    print(srcPath + "'s resolution is " + str(im.pixel_array.shape))
    im = im.pixel_array.astype(float)
## anti aliasing is true. It is resized into 800 * 600. 
    im = resize(im, (im_dim_x, im_dim_y), anti_aliasing=True)

## Rescaled into gray scale image. 
    rescaled_image = (np.maximum(im, 0)/im.max())*65536

    final_image= np.uint16(rescaled_image)

    print(srcPath + "'s resolution rescaled to " + str(final_image.shape))


    final_image = Image.fromarray(final_image)
    final_image.save(dstFolderPath)


# In[6]:


PositiveCBISDF = []
NegativeCBISDF = []
### Get labels for each mammogram and copy them into a separate local folder and create their metadata files. 
def RegenerateUIDandPath(dataset):
    for idx in dataset.index:
        lastChar = dataset['Subject ID'][idx][-1]
        if lastChar.isnumeric():
            print("the last character is numeric. ANd is going to be continued!")
            continue
        
        distinctiveName = dataset['Subject ID'][idx]
        distinctiveUID = dataset['Series UID'][idx]
        distinctiveRow = dataset.iloc[idx]
        ## 1-1 is the breast image
        RawPath = dataset['File Location'][idx] + "//1-1.dcm"
        ## File path is changed to the local file path
        NewPathSrc = "/home/careinfolab/StanfordMammogram" + RawPath[1:len(RawPath)]
        
        NewPositivePath = "/home/careinfolab/FIR_Inchan/breast_CBISDDSM/images/800x600/benign_negative/pos/"
        NewNegativePath = "/home/careinfolab/FIR_Inchan/breast_CBISDDSM/images/800x600/benign_negative/neg/"
        if not os.path.exists(NewPositivePath):
        # if the demo_folder directory is not present
        # then create it.
            os.makedirs(NewPositivePath)
            print("Positive path is created!")
            
        if not os.path.exists(NewNegativePath):
        # if the demo_folder directory is not present
        # then create it.
            os.makedirs(NewNegativePath)
            print("Negative path is created!")
        
        label = MalignOrBenign(distinctiveName)
        print(NewPathSrc + " is : " + label)
        if 'benign' in label.lower():
            destFolderPath =  NewNegativePath + str(idx) + ".png"
            NegativeCBISDF.append([distinctiveUID, destFolderPath])
            print("Benign :" + str(destFolderPath))
            print('label :' + label)
            
        else:
            destFolderPath =  NewPositivePath + str(idx) + ".png"
            PositiveCBISDF.append([distinctiveUID, destFolderPath])
            print("Malignant : " + str(destFolderPath))
            print('label :' + label)
        ConvertFromDCMtoPNG(NewPathSrc, destFolderPath)


# In[7]:


RegenerateUIDandPath(cbisddsm)


# In[9]:


neg_df = pd.DataFrame(NegativeCBISDF, columns=['empi_anon', 'file_path'])
pos_df = pd.DataFrame(PositiveCBISDF, columns=['empi_anon', 'file_path'])


# In[10]:


###Metadata files are created for benign and malignant mammogram images.
neg_df.to_csv('/home/careinfolab/FIR_Inchan/breast_CBISDDSM/images/800x600/benign_negative/neg_empi_path.csv')
pos_df.to_csv('/home/careinfolab/FIR_Inchan/breast_CBISDDSM/images/800x600/benign_negative/pos_empi_path.csv')


# In[ ]:




