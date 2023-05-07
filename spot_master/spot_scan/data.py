import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from joblib import dump
from sklearn.svm import SVC 

def extract_patches(img_path, csv_path, patch_size=5):
    """
    According to the coordinates from df.
    From img extracting patch_size * patch_size patches for training model.
    
    """   
    img_array = []
    label = []
    for filename in os.listdir(img_path):
        
        base = filename.split(".")
        img_ = Image.open(f"{img_path}/{base[0]}.tif")
        img = np.array(img_)
        df = pd.read_csv(f"{csv_path}/{base[0]}.csv")
        
        for index,row in df.iterrows():
            y,x = float(row[0]),float(row[1])
            patch = img[int(y)-patch_size//2:int(y)+patch_size//2+1,
                        int(x)-patch_size//2:int(x)+patch_size//2+1]
            if patch.shape == (patch_size,patch_size):
                img_array.append(patch.ravel())
                label.append(1)

        m = 1
        for index,row in df.iterrows(): 
            while m < 500:
                rand_y = np.random.randint(patch_size//2,img.shape[0]-patch_size//2-1)
                rand_x = np.random.randint(patch_size//2,img.shape[1]-patch_size//2-1)
                if abs(rand_x - int(x)) > patch_size//2 or abs(rand_y - int(y)) > patch_size//2:
                    neg_patch = img[rand_y-patch_size//2:rand_y+patch_size//2+1,
                                    rand_x-patch_size//2:rand_x+patch_size//2+1]
                    if neg_patch.shape==(patch_size,patch_size):
                        img_array.append(neg_patch.ravel())
                        label.append(0)
                        m+=1
    
    return img_array, label

 

def split_patches(img_array,label):
    
    x_train, x_test, y_train, y_test = train_test_split(img_array, label, test_size=0.4, random_state=50)
    
    return x_train, x_test, y_train, y_test


def standardization(x_train,x_test):
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    pickle.dump(scaler,open('scaler.pkl','wb'))
    
    return x_train_scaled,x_test_scaled,scaler

def train_model(x_train_scaled, y_train):
    
    clf = SVC(kernel='rbf',C=1,gamma=1,probability=True)
    clf.fit(x_train_scaled, y_train)
    dump(clf, 'svm_model.joblib')
    
    return clf
