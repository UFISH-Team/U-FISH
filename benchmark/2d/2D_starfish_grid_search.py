""" 2D grid search script for starfish BlobDetector parameters """

import os
from skimage import io
import pandas as pd
from starfish.core.spots.FindSpots import BlobDetector
from multiprocessing import Pool

import time

# Define a function to process each image
def process_im(im_path, out_path, sig_range, thr_range):
    im = io.imread(im_path)
    
    for sig in sig_range:
        for thr in thr_range:
            detector = BlobDetector(
                min_sigma=1.0,
                max_sigma=sig,
                num_sigma=sig,  
                threshold=thr,
                measurement_type='mean',   
                is_volume=False,
            )
            
            spots = detector.image_to_spots(im).spot_attrs

            df_path = os.path.join(out_path, f'ST_{str(os.path.basename(im_path))}
                                   _sig{sig}_thr{thr}.csv')
            df = spots.data
            df.rename(columns={'y': 'axis-0', 'x': 'axis-1'}, inplace=True)
            df1 = df[['axis-0', 'axis-1']] 

            df1.to_csv(df_path, index=False)
            print(f'Finished {df_path}', flush=True)

# path = /path/to/your/images/
im_path = 'ufish/valid/image/'

# path = /path/to/your/output/
out_path = 'ufish/valid/ST_results/'

# Parameters for grid search
sig_range = range(1, 6, 1)

thr_range = []
thr = 0.000095
while thr <= 0.15:
    thr_range.append(thr)
    thr += 0.00005

# Get a list of image files
image_files = [os.path.join(im_path, im) for im in os.listdir(im_path) if im.endswith('.tif') 
               and os.path.isfile(os.path.join(im_path, im))]

start_time = time.time()   
# Create a Pool for parallel processing
with Pool(processes=70) as pool:  # You can adjust the number of processes as needed
    pool.starmap(process_im, [(im_file, out_path, sig_range, thr_range) for im_file in image_files])

end_time = time.time()-start_time # in seconds
print(f'Total time: {end_time}')



 

