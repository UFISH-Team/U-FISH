""" 
     Test the complete process directly  
     Input: original image
     Output: .csv

"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.morphology import white_tophat
from skimage.feature import peak_local_max

def local_max(img,q): 
    t = white_tophat(img)
    spots = np.array(peak_local_max(t,threshold_abs=q,min_distance=2)) #q is threshold_abs
    return spots

def create_postive_patches(img,df):
    '''
    img: np.array(from .tif)
    df: coordinates(from .csv)
    patch: 1d
    '''
    pos_patches = []
    height, width = img.shape  
    for index,row in df.iterrows():
        y,x = row[0],row[1]                    
        if x < 2 or y < 2 or x > width - 3 or y > height - 3:           
            top, bottom = max(2 - y, 0), max(y - height + 3, 0)
            left, right = max(2 - x, 0), max(x - width + 3, 0)
            patch = np.pad(img[max(y-2,0):y+3, max(x-2,0):x+3], ((top, bottom), (left, right)), mode='constant')
            patch = patch.ravel()
        else:  
            patch = img[y-2:y+3, x-2:x+3]
            patch = patch.ravel()
            
        pos_patches.append([patch,y,x])
        
    return pos_patches

valid_path = "data/valid/187/x_test"
list_valid = os.listdir(valid_path)

# Load standard parameter and model
scaler = pickle.load(open('scaler_2.pkl','rb'))
model = joblib.load('svm_model_2.joblib')

predict_path = './data/valid/187/y_predict_2'
if not os.path.exists(predict_path):
    os.makedirs(predict_path)
    
# calling spots flow
for img in list_valid:
    
    base = img.split('.')
    base_ = base[0].split('_')
    
    img = np.array(Image.open(f'{valid_path}/{base[0]}.tif'))
    df = pd.DataFrame(local_max(img,15))
    pos_ =create_postive_patches(img,df) # a list
   
     
    pos_locations = []
    for matrix in pos_:   
        pos_scaled = scaler.transform(matrix[:-2])
        predict = model.predict(pos_scaled)
        if predict == 1:
            y_index = matrix[-2]
            x_index = matrix[-1]
            pos_locations.append([y_index, x_index])
            df = pd.DataFrame(pos_locations)
            df.columns=['axis-0','axis-1']  # napari format
            df.to_csv(f"{path}/{base[0]}.csv",index = False)
