#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd

import numpy as np
import os
import random
import shutil
from sklearn.utils import shuffle


# In[13]:


## Read metadata files. This step is similar to the stage 2. 
negativeimg = pd.read_csv('/home/careinfolab/FIR_Inchan/breast_simple_comparison/images/800x600/br12_456/neg_empi_path.csv')
positiveimg = pd.read_csv('/home/careinfolab/FIR_Inchan/breast_simple_comparison/images/800x600/br12_456/pos_empi_path.csv')

negativeimg = shuffle(negativeimg, random_state=5)
positiveimg = shuffle(positiveimg, random_state=5)


# In[14]:


empi_anon_pos = shuffle(list(positiveimg['empi_anon'].unique()), random_state=5)
empi_anon_neg = shuffle(list(negativeimg['empi_anon'].unique()), random_state=5)
TrainSize = 0.6 * len(positiveimg)
TestValSize = 0.2 * len(positiveimg) 
print(len(empi_anon_pos))
patient_id_df_pos = pd.DataFrame(empi_anon_pos, columns=['empi_anon'])
patient_id_df_neg = pd.DataFrame(empi_anon_neg, columns=['empi_anon'])


# In[15]:


## Patient IDs are stratified across test, val, train sets

trainset_pos = []
testset_pos = []
valset_pos = []
train_empi_anon_pos = []
test_empi_anon_pos = []
val_empi_anon_pos = []
for idx, anon in enumerate(empi_anon_pos):
    if idx == 0:
        trainset_pos = positiveimg[positiveimg['empi_anon'] == anon]
        
    
    if len(trainset_pos)>= TrainSize:
        print("Train : " + str(len(trainset_pos)))
        print("Break out")
        train_empi_anon_pos = [*set(train_empi_anon_pos)]
        break
        
    temp = positiveimg[positiveimg['empi_anon'] == anon]
    trainset_pos = trainset_pos.append(temp, ignore_index = True)
    train_empi_anon_pos.append(anon)
    
PatientID_Val_Test = patient_id_df_pos[~patient_id_df_pos.empi_anon.isin(train_empi_anon_pos)]

empi_anon_val = shuffle(list(PatientID_Val_Test['empi_anon'].unique()), random_state=4)
for idx, anon in enumerate(empi_anon_val):
    if idx == 0:
        valset_pos = positiveimg[positiveimg['empi_anon'] == anon]
        
    
    if len(valset_pos)>= TestValSize:
        print("Val : "+str(len(valset_pos)))
        print("Val Break out")
        val_empi_anon_pos = [*set(val_empi_anon_pos)]
        break
        
    temp = positiveimg[positiveimg['empi_anon'] == anon]
    valset_pos = valset_pos.append(temp, ignore_index = True)
    val_empi_anon_pos.append(anon)

PatientID_Test = PatientID_Val_Test[~PatientID_Val_Test.empi_anon.isin(val_empi_anon_pos)]

empi_anon_test = shuffle(list(PatientID_Test['empi_anon'].unique()), random_state=4)

for idx, anon in enumerate(empi_anon_test):
    if idx == 0:
        testset_pos = positiveimg[positiveimg['empi_anon'] == anon]

    temp = positiveimg[positiveimg['empi_anon'] == anon]
    testset_pos = testset_pos.append(temp, ignore_index = True)
    test_empi_anon_pos.append(anon)
    
    if idx == (len(empi_anon_test) - 1):
        print("Test : "+str(len(testset_pos)))
        test_empi_anon_pos = [*set(test_empi_anon_pos)]


# In[16]:


trainset_neg = []
testset_neg= []
valset_neg = []
train_empi_anon_neg = []
test_empi_anon_neg = []
val_empi_anon_neg = []
for idx, anon in enumerate(empi_anon_neg):
    if idx == 0:
        trainset_neg = negativeimg[negativeimg['empi_anon'] == anon]
        
    
    if len(trainset_neg)>= TrainSize:
        print("Train : " + str(len(trainset_neg)))
        print("Break out")
        train_empi_anon_neg = [*set(train_empi_anon_neg)]
        break
        
    temp = negativeimg[negativeimg['empi_anon'] == anon]
    trainset_neg = trainset_neg.append(temp, ignore_index = True)
    train_empi_anon_neg.append(anon)
    
PatientID_Val_Test = patient_id_df_neg[~patient_id_df_neg.empi_anon.isin(train_empi_anon_neg)]

empi_anon_val = shuffle(list(PatientID_Val_Test['empi_anon'].unique()), random_state=4)
for idx, anon in enumerate(empi_anon_val):
    if idx == 0:
        valset_neg = negativeimg[negativeimg['empi_anon'] == anon]
        
    
    if len(valset_neg)>= TestValSize:
        print("Val : "+str(len(valset_neg)))
        print("Val Break out")
        val_empi_anon_neg = [*set(val_empi_anon_neg)]
        break
        
    temp = negativeimg[negativeimg['empi_anon'] == anon]
    valset_neg = valset_neg.append(temp, ignore_index = True)
    val_empi_anon_neg.append(anon)

PatientID_Test = PatientID_Val_Test[~PatientID_Val_Test.empi_anon.isin(val_empi_anon_neg)]

empi_anon_test = shuffle(list(PatientID_Test['empi_anon'].unique()), random_state=4)

for idx, anon in enumerate(empi_anon_test):
    if idx == 0:
        testset_neg = negativeimg[negativeimg['empi_anon'] == anon]

    if len(testset_neg)>= TestValSize:
        print("Test : "+str(len(testset_neg)))
        print("Test Break out")
        test_empi_anon_neg = [*set(test_empi_anon_neg)]
        break
        
    temp = negativeimg[negativeimg['empi_anon'] == anon]
    testset_neg = testset_neg.append(temp, ignore_index = True)
    test_empi_anon_neg.append(anon)


# In[17]:


def SplitJPGtoSeperateFolder(TrainMal, TestMal, ValMal, TrainNeg, TestNeg, ValNeg, suffix = '_800_600_combined_comparison_birad'):


    if not os.path.exists('train' + suffix):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs("train" + suffix)
        os.makedirs("train" + suffix + "/pos/")
        os.makedirs("train" + suffix + "/neg/")

    if not os.path.exists("test" + suffix):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs("test" + suffix)
        os.makedirs("test"+ suffix+"/pos/")
        os.makedirs("test"+ suffix+"/neg/")

    if not os.path.exists("val" + suffix):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs("val" + suffix)
        os.makedirs("val"+ suffix+"/pos/")
        os.makedirs("val"+ suffix+"/neg/")

    for index in TrainMal.index:
        srcPath = TrainMal['file_path'][index]
#         ConvertJPGResolutionTOFixedResolution(srcPath, "./train"+ suffix+"/pos")
        shutil.copy(srcPath, "./train"+ suffix+"/pos") 
        # shutil.copy2(srcPath, "./train/neg")
        print(srcPath + " has been copied(train positive)")



    for index in TestMal.index:
        srcPath = TestMal['file_path'][index]
#         ConvertJPGResolutionTOFixedResolution(srcPath, './test'+ suffix+'/pos')
        shutil.copy(srcPath, './test'+ suffix+'/pos')
        # shutil.copy2(srcPath, './test/neg')
        print(srcPath + " has been copied(test positive)")

    for index in ValMal.index:
        srcPath = ValMal['file_path'][index]
#         ConvertJPGResolutionTOFixedResolution(srcPath, './val'+ suffix+'/pos')
        shutil.copy(srcPath, './val'+ suffix+'/pos')
        # shutil.copy2(srcPath, './test/pos')
        print(srcPath + ' has been copied(val positive)')

    for index in TrainNeg.index:
        srcPath = TrainNeg['file_path'][index]
#         ConvertJPGResolutionTOFixedResolution(srcPath, './train'+ suffix+'/neg')
        shutil.copy(srcPath, './train'+ suffix+'/neg')
        # shutil.copy2(srcPath, './val/neg')
        print(srcPath + ' has been copied(train negative)')

    for index in TestNeg.index:
        srcPath = TestNeg['file_path'][index]

#         ConvertJPGResolutionTOFixedResolution(srcPath, './test'+ suffix+'/neg')
        shutil.copy(srcPath, './test'+ suffix+'/neg')
        # shutil.copy2(srcPath, './val/pos')
        print(srcPath + ' has been copied(test negative)')

    for index in ValNeg.index:
        srcPath = ValNeg['file_path'][index]
#         ConvertJPGResolutionTOFixedResolution(srcPath, './val'+ suffix+'/neg')
        shutil.copy(srcPath, './val'+ suffix+'/neg')
        # shutil.copy2(srcPath, './val/pos')
        print(srcPath + ' has been copied(validation negative)')
        
        


SplitJPGtoSeperateFolder(trainset_pos, testset_pos, valset_pos, trainset_neg, testset_neg, valset_neg)


# In[18]:


cbisnegativeimg = pd.read_csv('/home/careinfolab/FIR_Inchan/breast_CBISDDSM/images/800x600/benign_negative/neg_empi_path.csv')
cbispositiveimg = pd.read_csv('/home/careinfolab/FIR_Inchan/breast_CBISDDSM/images/800x600/benign_negative/pos_empi_path.csv')

cbisnegativeimg = shuffle(cbisnegativeimg, random_state=5)
cbispositiveimg = shuffle(cbispositiveimg, random_state=5)


# In[19]:


## Size setup
TrainSize = len(cbispositiveimg) * 0.6
ValSize = len(cbispositiveimg) * 0.2
TestSize = len(cbispositiveimg) * 0.2


# In[20]:


def SplitJPGtoSeperateFolderNoTest(TrainMal, TestMal, ValMal, TrainNeg, TestNeg, ValNeg, suffix = '_800_600_combined_comparison_birad'):


    if not os.path.exists('train' + suffix):
        os.makedirs("train" + suffix)
        os.makedirs("train" + suffix + "/pos/")
        os.makedirs("train" + suffix + "/neg/")

    if not os.path.exists("test" + suffix):
        os.makedirs("test" + suffix)
        os.makedirs("test"+ suffix+"/pos/")
        os.makedirs("test"+ suffix+"/neg/")

    if not os.path.exists("val" + suffix):
        os.makedirs("val" + suffix)
        os.makedirs("val"+ suffix+"/pos/")
        os.makedirs("val"+ suffix+"/neg/")

    for index in TrainMal.index:
        srcPath = TrainMal['file_path'][index]
        shutil.copy(srcPath, "./train"+ suffix+"/pos") 
        print(srcPath + " has been copied(train positive)")





    for index in ValMal.index:
        srcPath = ValMal['file_path'][index]
        shutil.copy(srcPath, './val'+ suffix+'/pos')
        print(srcPath + ' has been copied(val positive)')

    for index in TrainNeg.index:
        srcPath = TrainNeg['file_path'][index]
        shutil.copy(srcPath, './train'+ suffix+'/neg')

        print(srcPath + ' has been copied(train negative)')



    for index in ValNeg.index:
        srcPath = ValNeg['file_path'][index]
        shutil.copy(srcPath, './val'+ suffix+'/neg')
        print(srcPath + ' has been copied(validation negative)')
        
        



# In[21]:


def SplitMALIGNMENT_BENIGN(neg_dataset, pos_dataset):
    # train = dataset.sample(frac=0.6, random_state=200)  # random state is a seed value
    # test = dataset.drop(train.index)
    # val = test.sample(frac = 0.5, random_state=200)
    # test = test.drop(val.index)

# 	## I used 60:20:20 ratio for both CBIS-DDSM, and EMBED
#     malignant = dataset.groupby('Label').get_group('MALIGNANT')
#     others = dataset[dataset["Label"] != 'MALIGNANT']

    trainMal = pos_dataset.sample(frac=0.6, random_state=200)
    testMal = pos_dataset.drop(trainMal.index)
    valMal = testMal.sample(frac = 0.5, random_state = 200)
    testMal = testMal.drop(valMal.index)

    trainOthers = neg_dataset.sample(frac = 0.6, random_state = 200)
    testOthers = neg_dataset.drop(trainOthers.index)
    valOthers = testOthers.sample(frac = 0.5, random_state = 200)
    testOthers = testOthers.drop(valOthers.index)

    return trainMal, testMal, valMal, trainOthers, testOthers, valOthers

trainMal, testMal, valMal, trainOthers, testOthers, valOthers = SplitMALIGNMENT_BENIGN(cbisnegativeimg, cbispositiveimg)
SplitJPGtoSeperateFolderNoTest(trainMal, testMal, valMal, trainOthers, testOthers, valOthers)


# In[ ]:





# In[ ]:




