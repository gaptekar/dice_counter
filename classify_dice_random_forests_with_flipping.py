# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:29:18 2017

@author: Gabriel
"""

from os import listdir
from skimage.io import imread, imsave
import numpy as np
from skimage.transform import resize, rescale, rotate
import pickle
from skimage import img_as_ubyte, img_as_float
from random import shuffle
#from image_util import show_image
images=[]
add_rotations=True
for folder in "123456":
    grouped_images=[]
    file_names=listdir(folder+"/cleaned")
    shuffle(images)
    for i, file in enumerate(file_names):
        if i==50:break
        original_image=imread(folder+"/cleaned/"+file)
        original_image=img_as_ubyte(original_image)    
        if add_rotations:
            for deg in range(-180,180,5):
                image=rotate(original_image,deg)
                image=image.flatten()
                grouped_images.append(image)
        else:
            image=original_image.flatten()
            grouped_images.append(image)         
    images.append(np.vstack(grouped_images))

 

#show_image(a[0,:].reshape(50,50))
labels=[]
for i,group_of_images in enumerate(images):
    labels.append(np.ones(group_of_images.shape[0])*(i+1))
labels=np.hstack(labels).astype(int)
images=np.vstack(images)


pickle.dump(images,open( "test_images_rotations.pkl", "wb" ))
pickle.dump(labels,open( "test_labels_rotations.pkl", "wb" ))



#%%
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 



pipe = Pipeline([("rf",RandomForestClassifier()) ])


#%%
param_range=[10,20,30]
param_grid = [{"rf__n_estimators":param_range,
               "rf__min_samples_split":[100]}]
               
grid_search = GridSearchCV( estimator=pipe,
                            param_grid=param_grid,
                            scoring="accuracy",
                            cv=3,
                            n_jobs=1,
                            verbose=10)
 
grid_search.fit(images,labels) 
print(f"The best score was: {grid_search.best_score_}")
print(grid_search.best_params_)   
    
pickle.dump(grid_search.best_estimator_,open( "model_rf.pkl", "wb" ))

#%%

