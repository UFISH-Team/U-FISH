import subprocess
import os

sizes = [0, 1, 2]
shapes = ["disk", "diamond"]

for size in sizes:
    for shape in shapes:
        target_process = f"{shape}({size})"
        target_dir = f"./spotlearn_metrics/{target_process}"

        os.makedirs(target_dir, exist_ok=True)

        ufish_command = (f"/home/hycai/.conda/envs/spotlearn/bin/python  -m  ufish - set_logger ./spotlearn_metrics/"$target_process"/spot_learn_b2_2.log - init-model --model-type 'spot_learn' - train ../meta_train.csv ../meta_valid.csv --root-dir ./FISH_spots --model_save_dir ./spotlearn_metrics/"$target_process" --num_epochs 150 --batch_size 2 --loader-workers=4 --loss-type 'DiceRMSELoss' --lr 1e-5 --target-process "${target_process}"

")

        print(f"Running command for target process: {target_process}")
        print(f"Command: {ufish_command}")

        subprocess.call(ufish_command, shell=True)