"""Trains and evaluates the random forest classifier for scratch segmentation"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from feature_extraction import calc_features, addTruthLabel

# Training and their ground truths image path and files
imgpath = r"C:\Users\shubh\Documents\steel_detection\metal_nut\test\scratch"
gdpath = r"C:\Users\shubh\Documents\steel_detection\metal_nut\ground_truth\scratch"

imglist = ["007.png", "011.png", "014.png", "015.png", "016.png", "017.png"]
gdlist = ["007_mask.png", "011_mask.png", "014_mask.png", "015_mask.png", "016_mask.png", "017_mask.png"]

imgfiles = [os.path.join(imgpath,file) for file in imglist]
gdfiles = [os.path.join(gdpath,file) for file in gdlist]

# Adding all the features of different images in one dataframe
df = pd.DataFrame([])
for i in range(len(imgfiles)):
    if df.empty:
        df = calc_features(imgfiles[i])
        df = addTruthLabel(df,gdfiles[i])
    else:
        dftemp = calc_features(imgfiles[i])
        dftemp = addTruthLabel(dftemp,gdfiles[i])
        df = pd.concat([df,dftemp])

# print(df.head())

# Separating independent and dependent variables in X and Y
Y = df['ground_truth'].values
X = df.drop('ground_truth', axis=1)

# Splitting training and validation sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=20)

# Random forest classifier model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, Y_train)

# Predicting truths for validation set
test_pred = model.predict(X_test)
# print(train_pred)

# Finding accuracy of model on validation dataset
print("Accuracy: ", metrics.accuracy_score(Y_test, test_pred))

# Finding importance of different features of model
# Features can be used to optimize model and remove those features
# which are not importance for model
importance = list(model.feature_importances_)
features_list = list(df.columns)

# Dumping model into to binary file
filename = 'scratch_detect_model'
modelpath = os.path.join(os.getcwd(),"model")
if not os.path.isdir(modelpath):
    os.mkdir(modelpath)

pickle.dump(model, open(os.path.join(modelpath,filename), 'wb'))

#load_model = pickle.load(open(filename,'rb'))