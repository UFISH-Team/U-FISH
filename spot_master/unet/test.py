import os
import torch
import pandas as pd

from model import UNet
from skimage.io import imread
from skimage.feature import peak_local_max
from f1score import compute_metrics


model = UNet(1,1,4)
model.load_state_dict(torch.load('./best_unet_model_after_fine_tuning.pth', map_location=torch.device('cpu')))

root_dir = '/home/hycai/pytorch/SpotMaster-main/spot_master/FISH_spots/'
meta_data = pd.read_csv('/home/hycai/pytorch/SpotMaster-main/notebooks/meta_test.csv')

img_dir = meta_data.iloc[:, 0].tolist()
marker_dir = meta_data.iloc[:, 1].tolist()
predict_dir = '/home/hycai/pytorch/SpotMaster-main/spot_master/unet/predict/'

def pred(im):
    im_t = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    p = model(im_t)
    return p.squeeze(0).squeeze(0).detach().numpy()

def get_coordinates(image_path):
    image = imread(image_path)
    predict = pred(image)
    out_1 = predict / predict.max()
    coordinates = peak_local_max(out_1, min_distance=1, threshold_rel=0.5)
    return coordinates

for img in img_dir:
    coordinates = get_coordinates(os.path.join(root_dir, img))
    coordinates_df = pd.DataFrame(coordinates)
    basename = os.path.basename(img)
    output_name = os.path.splitext(basename)
    coordinates_df.to_csv(os.path.join(predict_dir, output_name[0]) + '.csv', index=False, header=['axis-0', 'axis-1'])

for file in marker_dir:
    basename = os.path.basename(file)
    df1 = pd.read_csv(os.path.join(predict_dir, basename))
    df2 = pd.read_csv(os.path.join(root_dir, file))
    np1 = df1.values
    np2 = df2.values
    f1score = compute_metrics(np1, np2, 3.0)
    cols_to_save = ['cutoff', 'f1_score']
    f1score[cols_to_save].to_csv(f'./f1score/{basename}', index=False)

convert_data = {}
for filename in os.listdir('./f1score/'):  # 利用for循环逐个读取csv文件
    input_dir = './f1score/'
    df = pd.read_csv(os.path.join(input_dir, filename))
    convert_data[filename] = [df["f1_score"][49]] # TODO 这里取cutoff=0的第一行
all_dataframe = pd.DataFrame(convert_data)
all_dataframe.to_csv("f1score_final.csv", sep=',', index=0)

