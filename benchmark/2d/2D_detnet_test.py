import subprocess

alphas = [0.0, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889, 1.0]

for alpha in alphas:
    command = f".conda/envs/spotlearn/bin/python -m ufish init-model --model-type 'det_net' --alpha {alpha} load-weights ./{alpha}/best_model.pth -predict-imgs test_image ./predict/{alpha}"
    subprocess.call(command, shell=True)

import subprocess
import numpy as np

alphas = np.round(np.linspace(0, 1, 10), 4)

for alpha in alphas:
    command = f".conda/envs/spotlearn/bin/python -m ufish evaluate_imgs ./predict/{alpha} ./test_gt detnet-{alpha}.csv"
    
    subprocess.run(command, shell=True)
