""" The complete process of training  model """

import os 
import pickle
import numpy as np
from sklearn.svm import SVC  
from sklearn import metrics
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

import datetime

starttime = datetime.datetime.now()

# Download Data
datasets = ['train','test']
for i in datasets:
    datadir1 = f"data/{i}/rca-1"
    img1 = os.listdir(datadir1)
    datadir0 = f"data/{i}/rca-0"
    img0 = os.listdir(datadir0)

    image1 = []
    label1 = []
    for j in img1:
        img = np.array(Image.open(os.path.join(datadir1,j)),dtype=float).ravel()
        image1.append(img)
        label1.append(1)

    image0 = []
    label0 = []
    for j in img0:
        img = np.array(Image.open(os.path.join(datadir0,j)),dtype=float).ravel()
        image0.append(img)
        label0.append(0)  
    if i=="train":
        train_data = np.concatenate((image1,image0),axis=0)
        train_label = np.concatenate((label1,label0),axis=0)
    if i=="test":
        test_data = np.concatenate((image1,image0),axis=0)
        test_label = np.concatenate((label1,label0),axis=0)    

# Randomly assigned
x_train, x, y_train, y = train_test_split(train_data, train_label, test_size=0.001, random_state=50)
x_, x_test, y_, y_test = train_test_split(test_data, test_label, test_size=0.999, random_state=50)

# Standardization
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
pickle.dump(scaler,open('scaler_2.pkl','wb'))

# Train the SVM model Normalized data
clf = SVC(kernel='rbf',C=1,gamma=1,probability=True)
clf.fit(x_train_scaled, y_train)
dump(clf, 'svm_model_2.joblib')

# Predict test data
y_pred = clf.predict(x_test_scaled)
aftstd_acc = accuracy_score(y_test, y_pred)
print(f"aftstd_accuracy: {aftstd_acc}")

# Model evaluation
print(f"\n{metrics.classification_report(y_test, y_pred)}")

#Comparison of predicted results with actual results
print(f'test number: {len(y_test)}\npredict right number：{sum(y_pred==y_test)}')

endtime = datetime.datetime.now()
print (f'running time: {(endtime - starttime).seconds}s')
