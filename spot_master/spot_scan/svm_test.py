import os
from PIL import Image
import numpy as np
import pandas as pd

from predict import local_max,create_postive_patches


def test(img_path,scaler,model,predict_path,q):

    for filename in os.listdir(img_path):
    
        base = filename.split('.')
    
        img = np.array(Image.open(f'{img_path}/{base[0]}.tif'))
        df = local_max(img,q)
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
        df.to_csv(f"{predict_path}/{base[0]}.csv",index = False)

