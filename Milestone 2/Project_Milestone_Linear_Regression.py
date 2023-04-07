from sklearn import datasets, linear_model 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.applications import EfficientNetB3, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from PIL import Image
import cv2
np.random.seed(42)

# Load and preprocess data
sample_df = pd.read_csv('C:/Users/sahil/Downloads/cassava-leaf-disease-classification/train.csv')
test_image=cv2.imread('C:/Users/sahil/Downloads/cassava-leaf-disease-classification/test_images/2216849948.jpg')
cv2.imshow('image',test_image)
img_=cv2.resize(test_image, None, fx = 0.2, fy = 0.2)
image_test=img_.flatten()
	
# image_test = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
cv2.imshow('image',image_test)
print(image_test.shape)
sample_df['label'] = sample_df['label'].astype(str)
sample_df['path'] = sample_df['image_id'].apply(lambda x: 'C:/Users/sahil/Downloads/cassava-leaf-disease-classification/train_images/' + x)
sample_df = sample_df[['path', 'label']]
sample_df.head()
print(type(sample_df.loc[0].at["path"]))
a=(sample_df.loc[0].at["path"])
img = cv2.imread(a)
img=cv2.resize(img, None, fx = 0.2, fy = 0.2)
image=img.flatten()
# image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(image.shape)
print(image)
print(len(sample_df.index))

for i in range(1,len(sample_df.index)):
    a=(sample_df.loc[i].at["path"])
    # print(i,"location")
    img = cv2.imread(a)
    img=cv2.resize(img, None, fx = 0.2, fy = 0.2)
    # img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(i,"image")
    img1=img.flatten()
    # print(i,"flat")
    image=np.vstack((image,img1))
    # print(i,"stack")
    # print(image.shape)
    if(i%100==0):
        print(i)
print(image)
print(image.shape)

regr = LogisticRegression(random_state=0)
regr.fit(image, sample_df['label'])
y_pred = regr.predict(image_test.reshape(1, -1))
print("predection:",y_pred)
