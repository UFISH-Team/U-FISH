import os
from PIL import Image
from sklearn import metrics
from sklearn.metrics import accuracy_score
from data import extract_patches,split_patches,standardization,train_model

img_path = '../train_data/image'
csv_path = '../train_data/csv'

list_img = os.listdir(img_path)
list_csv = os.listdir(csv_path)

def train(img_path, csv_path, patch_size=5):
    
    img_array,label = extract_patches(img_path, csv_path, patch_size=5)
    
    x_train, x_test, y_train, y_test= split_patches(img_array,label)
    
    x_train_scaled,x_test_scaled,scaler = standardization(x_train,x_test)
    
    model = train_model(x_train_scaled, y_train)
      
    # Predict test data
    y_pred = model.predict(x_test_scaled)
    aftstd_acc = accuracy_score(y_test, y_pred)
    print(f"aftstd_accuracy: {aftstd_acc}")
    
    # Model evaluation
    print(f"\n{metrics.classification_report(y_test, y_pred)}")

    #Comparison of predicted results with actual results
    print(f'test number: {len(y_test)}\npredict right number：{sum(y_pred==y_test)}')