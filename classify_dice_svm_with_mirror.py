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
#from image_util import show_image
images=[]
add_rotations=True
for folder in "123456":
    grouped_images=[]
    for i, file in enumerate(listdir(folder+"/better_tests")):
        original_image=imread(folder+"/better_tests/"+file)
        original_image=img_as_ubyte(original_image)
        
        if add_rotations:
            for deg in [-5,0,5]:
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


#%% add in the flip
mirror_images=images.T.reshape(50,50,-1) 
_,_,samples=mirror_images.shape
for i in range(samples):
    mirror_images[:,:,i]=rotate(mirror_images[:,:,i],180)
mirror_images=mirror_images.reshape(50*50,-1).T
mirror_images=mirror_images[labels>1,:]
mirror_labels=labels[labels>1].copy()+6


labels=np.hstack([labels,mirror_labels])
images=np.vstack([images,mirror_images])


pickle.dump(images,open( "test_images_with_corrections_and_rotations.pkl", "wb" ))
pickle.dump(labels,open( "test_labels_with_corrections_and_rotations.pkl", "wb" ))



#%%
import pandas as pd
import seaborn as sns

"""
pca.fit(images)
images_pca=pca.transform(images)

data=np.hstack([images_pca,labels[:,np.newaxis]])
df=pd.DataFrame(data=data)
#df.columns=["x","y","digit"]
df.columns=df.columns.astype(str)
sns.lmplot("0","1",df,"2",fit_reg =False)
"""



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe_svc = Pipeline([("sc1",StandardScaler()),
                    ("pca",PCA()),
                    ("clf",SVC())
                    ])


#%%
param_range=[.001,.01,1.0,10.0]
param_grid = [{"clf__C":param_range,
               "clf__gamma":param_range,
               "clf__kernel":["rbf"],
               "pca__n_components":[10,12,14]}]
               
grid_search = GridSearchCV( estimator=pipe_svc,
                            param_grid=param_grid,
                            scoring="accuracy",
                            cv=3,
                            n_jobs=1,
                            verbose=10 )
 
grid_search.fit(images.astype(float),labels) 
print(f"The best score was: {grid_search.best_score_}")
print(grid_search.best_params_)   
    
pickle.dump(grid_search.best_estimator_,open( "model_with_360_degree_rotations.pkl", "wb" ))

#%%
best_pipe = Pipeline([("sc1",StandardScaler()),
                    ("pca",PCA(n_components=)),
                    ("clf",SVC(kernel="rbf",gamma=0.01,C=0.01))
                    ])
best_pipe.fit(images,labels)
print(best_pipe.score(images,labels)) 
pickle.dump(best_pipe,open( "model.pkl", "wb" ))
