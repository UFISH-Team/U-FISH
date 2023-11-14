import subprocess
import os
import numpy as np

alphas = np.round(np.linspace(0, 1, 10), 4)

for alpha in alphas:
    os.makedirs(f'./DetNet_metrics/{alpha}', exist_ok=True)
    command = f".conda/envs/spotlearn/bin/python -m ufish - set_logger ./DetNet_metrics/{alpha}/det_net_b2_1.log - init-model --model-type 'det_net' - train ../meta_train.csv ../meta_valid.csv --root-dir ./FISH_spots --model_save_dir ./DetNet_metrics/{alpha} --num_epochs 150 --batch_size 2 --loader-workers=4 --loss-type 'DicesoftLoss' --lr 1e-5 --target-process None --alpha {alpha}"
    try:
        subprocess.run(command, shell=True, check=True)

    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
