import os
import math
import time

objects = ['disk', 'diamond']

for object in objects:
    if object == 'diamond':
        s_values = [math.floor(((i * 2 + 1) ** 2 + 1) * 0.75) for i in range(1, 4)]
    else:
        s_values = [math.floor(((i * 2 + 1) ** 2 - (i * 2 - 1)) * 0.75) for i in range(1, 4)]
    for i, s in enumerate(s_values):
        directory_path = f"./predict/{object}-{i+1}"         
        command = f"/home/hycai/.conda/envs/ome/bin/python -m  ufish - init-model --model-type 'spot_learn' load-weights ./{object}-{i+1}/best_model.pth - predict-imgs test_image ./predict/{object}-{i+1} --spot_calling_method cc_center --cc_size_threshold {s}" 
        print(f"Inference time for {object}-{i}: {inference_time} seconds")

import subprocess

objects = ['disk', 'diamond']
i_values = ['1', '2', '3']

for obj in objects:
    for i_val in i_values:
        command = f"/home/hycai/.conda/envs/spotlearn/bin/python -m ufish evaluate_imgs ./predict/{obj}-{i_val} ./test_gt spotlearn-{obj}-{i_val}.csv"
        
        subprocess.run(command, shell=True)
