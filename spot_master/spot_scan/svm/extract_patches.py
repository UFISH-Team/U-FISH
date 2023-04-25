"""According to the coordinates from .csv, the images is extracted as 5*5 patches for training model"""

import os
import imageio
import numpy as np
import pandas as pd
from PIL import Image
import shutil

train = []  #input image name
test = []

# extract train big images
img_path = "data/RCA_low/RCA_low_image"
csv_path = "data/RCA_low/RCA_low_csv"

list_img = os.listdir(img_path)
list_csv = os.listdir(csv_path)

selected_train = list_img[::50]

train_path = "data/train/rca_512"
if not os.path.exists(train_path):
    os.makedirs(train_path)

for img in selected_train:
    base = img.split('.')
    csv = f"{base[0]}.csv"
    if csv in list_csv:
        train.append(base[0])
        file1_path = os.path.join(img_path,img)
        shutil.copy(file1_path,train_path)
        file2_path = os.path.join(csv_path,csv)
        shutil.copy(file2_path,train_path)
        
# extract test big images
selected_test = list_img[20::100]

test_path = "data/test/rca_512"
if not os.path.exists(test_path):
    os.makedirs(test_path)

for img in selected_test:
    base = img.split('.')
    csv = f"{base[0]}.csv"
    if csv in list_csv:
        test.append(base[0])
        file1_path = os.path.join(img_path,img)
        shutil.copy(file1_path,test_path)
        file2_path = os.path.join(csv_path,csv)
        shutil.copy(file2_path,test_path)
        
 # mkdir 
if not os.path.exists("data/train/rca-1"):
    os.makedirs("data/train/rca-1")
if not os.path.exists("data/train/rca-0"):
    os.makedirs("data/train/rca-0")
if not os.path.exists("data/test/rca-1"):
    os.makedirs("data/test/rca-1")
if not os.path.exists("data/test/rca-0"):
    os.makedirs("data/test/rca-0")
    
# train patches    
for i in train:   
    img = np.array(Image.open(f"{train_path}/{i}.tif"))
    df = pd.read_csv(f"{train_path}/{i}.csv")
    
    n = 1
    for index,row in df.iterrows():
        y,x = float(row[0]),float(row[1])
        patch = img[int(y)-2:int(y)+3,int(x)-2:int(x)+3]
        if patch.shape == (5,5):
            imageio.imwrite(f"./data/train/rca-1/{i}_1_{n}.tif",patch)
            n+=1
    m = 1
    for index,row in df.iterrows(): 
        while m <500:
            rand_y = np.random.randint(2,img.shape[0]-3)
            rand_x = np.random.randint(2,img.shape[1]-3)
            if abs(rand_x - int(x)) > 2 or abs(rand_y - int(y)) >2:
                neg_patch = img[rand_y-2:rand_y+3,rand_x-2:rand_x+3]
                if neg_patch.shape==(5,5):
                    imageio.imwrite(f"./data/train/rca-0/{i}_0_{m}.tif",neg_patch)
                    m+=1
# test patches
for i in test:
    img = np.array(Image.open(f"{test_path}/{i}.tif"))
    df = pd.read_csv(f"{test_path}/{i}.csv")
    
    a = 1
    for index,row in df.iterrows():
        y,x = float(row[0]),float(row[1])
        patch = img[int(y)-2:int(y)+3,int(x)-2:int(x)+3]
        if patch.shape == (5,5):
            imageio.imwrite(f"./data/test/rca-1/{i}_1_{a}.tif",patch)
            a+=1
    b=1
    for index,row in df.iterrows():  
        while b <500:
            rand_y = np.random.randint(2,img.shape[0]-3)
            rand_x = np.random.randint(2,img.shape[1]-3)
            if abs(rand_x - int(x)) > 2 or abs(rand_y - int(y)) >2:
                neg_patch = img[rand_y-2:rand_y+3,rand_x-2:rand_x+3]
                if neg_patch.shape==(5,5):
                    imageio.imwrite(f"./data/test/rca-0/{i}_0_{b}.tif",neg_patch)
                    b+=1   
